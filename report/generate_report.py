"""Build the English report and figures from archived records and validation diagnostics."""
import os
os.environ.setdefault("MPLCONFIGDIR","/tmp/mp2-report-matplotlib")
from pathlib import Path
import sys,json,re,textwrap,hashlib
from xml.sax.saxutils import escape
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak,
    KeepTogether,Preformatted,
)
from pypdf import PdfReader

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
D=json.loads((HERE/"results.json").read_text())
FIG=HERE/"figures"
FIG.mkdir(exist_ok=True)
NAVY="#17364a";TEAL="#147d79";BLUE="#37799f";GREY="#7f929e";GOLD="#c29b4a";LIGHT="#eef4f6"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,
 "axes.spines.top":False,"axes.spines.right":False,"axes.labelcolor":NAVY,
 "text.color":NAVY,"axes.edgecolor":"#b7c5cc","xtick.color":NAVY,"ytick.color":NAVY,
 "savefig.facecolor":"white","svg.fonttype":"none"})
def savefig(fig,name):
    fig.savefig(FIG/(name+".png"),dpi=240,bbox_inches="tight",pad_inches=.08)
    fig.savefig(FIG/(name+".svg"),bbox_inches="tight",pad_inches=.08)
    plt.close(fig)
def architecture():
    fig=plt.figure(figsize=(7.2,3.25))
    ax=fig.add_axes([.01,.01,.98,.98])
    ax.set_xlim(0,10.5);ax.set_ylim(0,4.35);ax.axis("off")
    def box(x,y,w,h,title,detail,fc=LIGHT,edge="#b5cbd2"):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.06,rounding_size=.10",
                      facecolor=fc,edgecolor=edge,linewidth=.8))
        ax.text(x+w/2,y+h*.71,title,ha="center",va="center",weight="bold",fontsize=8.2)
        ax.text(x+w/2,y+h*.30,detail,ha="center",va="center",fontsize=6.9,linespacing=1.35)
    def arrow(a,b,dashed=False):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=9,
           color=GREY if dashed else NAVY,linewidth=1,linestyle="--" if dashed else "-"))
    box(.2,3.2,3.05,.85,"Given inputs","Question + 4 candidate texts\nFixed embeddings: 4 x 1024")
    box(3.75,3.2,3.05,.85,"Canonical answer groups","Official numeric parser\nSame final answer -> same group")
    arrow((3.31,3.62),(3.68,3.62))
    ax.text(8.8,3.77,"Question-local computation",ha="center",va="center",fontsize=8.2,weight="bold",color=TEAL)
    ax.text(8.8,3.40,"No reference access\nSame frozen method for new banks",ha="center",va="center",fontsize=6.9,linespacing=1.4)
    for x,w,title,detail in [
        (.2,3.05,"2,048 embedding columns","Group mean + mean x group context"),
        (3.75,3.05,"40 existing scalar slots","Votes, similarity, math/text evidence"),
        (7.3,3.0,"24 new lexical/numeric columns","12 candidate signals: mean and max")]:
        box(x,1.72,w,.85,title,detail)
    for x in (1.72,5.27,8.8):arrow((5.27,3.14),(x,2.63))
    box(.2,.15,3.05,.95,"Training-only supervision","Five-fold train CV; seed 7506\nL-BFGS-B; L2 = 0.03",fc="#faf1df",edge=GOLD)
    box(3.75,.15,3.05,.95,"Frozen linear scorer","Concat 2,112 columns; standardize + clip\ns(group) = w^T z(group)",fc="#dcefeb",edge=TEAL)
    box(7.3,.15,3.0,.95,"Return a supplied candidate","Highest-scoring answer group\nStable representative index")
    arrow((1.72,1.67),(5.27,1.16));arrow((5.27,1.67),(5.27,1.16));arrow((8.8,1.67),(5.27,1.16))
    arrow((3.31,.62),(3.68,.62),True);arrow((6.87,.62),(7.23,.62))
    savefig(fig,"architecture")
def performance():
    keys=list(D["models"]);names=[D["models"][k]["name"] for k in keys]
    fig,axes=plt.subplots(1,2,figsize=(9.5,3.1),sharey=True)
    yy=np.arange(len(keys))
    for ax,field,title in zip(axes,["val_accuracy","test"],["Validation: all 798 questions","Test: all 467 questions"]):
        vals=[D["models"][k][field]["accuracy"] if field=="test" else D["models"][k][field] for k in keys]
        ax.barh(yy,vals,color=[TEAL if k=="final" else "#9aaeb9" for k in keys],height=.61)
        ax.set_xlim(0,109);ax.set_xticks([0,25,50,75,100]);ax.set_xlabel("Selection accuracy (%)")
        ax.set_title(title,fontsize=10,loc="left",pad=10)
        for y,v in zip(yy,vals):ax.text(v+1,y,f"{v:.2f}",va="center",fontsize=8.5)
        ax.grid(axis="x",alpha=.15);ax.set_axisbelow(True)
    axes[0].set_yticks(yy,names,fontsize=8.5);axes[0].invert_yaxis()
    fig.tight_layout(w_pad=2)
    savefig(fig,"performance")
def ablation_plot():
    labels={"lexical":"Lexical (+24)","trace":"Trace (+24)","combined":"Combined (+48)","contextual":"Contextual (+80)"}
    fig,ax=plt.subplots(figsize=(8.6,2.45))
    colorset={"lexical":TEAL,"trace":GREY,"combined":BLUE,"contextual":GOLD}
    for r in D["l2_grid"]:
        schema=r["schema"]
        ax.plot([v["l2"] for v in r["values"]],[v["accuracy"] for v in r["values"]],
                marker="o",lw=1.7,color=colorset[schema],label=labels[schema],markersize=4)
    ax.set_xscale("log");ax.set_xticks([.01,.03,.1],["0.01","0.03","0.10"])
    ax.set_ylim(81.8,83.65);ax.set_ylabel("Train OOF accuracy (%)")
    ax.set_xlabel("Linear L2 penalty");ax.grid(alpha=.18)
    ax.legend(ncol=2,fontsize=8,loc="lower left",frameon=False)
    fig.tight_layout()
    savefig(fig,"l2_ablation")
def calibration_plot():
    bins=D["calibration"]["bins"]
    fig,axes=plt.subplots(1,2,figsize=(9.0,2.7),gridspec_kw={"width_ratios":[1,1.1]})
    x=[r["confidence"] for r in bins];y=[r["accuracy"] for r in bins]
    axes[0].plot([0,1],[0,1],ls="--",lw=1,color=GREY,label="Perfect calibration")
    axes[0].plot(x,y,"o-",color=TEAL,lw=1.5)
    for a,b,r in zip(x,y,bins):
        axes[0].annotate(f"n={r['n']}",(a,b),xytext=(-10,8),textcoords="offset points",fontsize=7.5)
    axes[0].set(xlim=(.25,1.02),ylim=(.25,1.05),xlabel="Mean selected-group probability",ylabel="Observed accuracy")
    axes[0].set_title("Descriptive reliability",loc="left",fontsize=10)
    axes[0].grid(alpha=.13)
    labels=[f"{r['low']:.1f}-{r['high']:.1f}" for r in bins]
    axes[1].bar(labels,[r["n"] for r in bins],color="#8bbab5",width=.62)
    for i,r in enumerate(bins):axes[1].text(i,r["n"]+3,str(r["n"]),ha="center",fontsize=8)
    axes[1].set_ylim(0,205);axes[1].set_ylabel("Questions");axes[1].set_xlabel("Selected-group probability bin")
    axes[1].set_title("355 validation conflict questions",loc="left",fontsize=10)
    axes[1].tick_params(axis="x",labelsize=8)
    fig.tight_layout(w_pad=2)
    savefig(fig,"validation_calibration")
def equations():
    equations=[
       r"$u_j=e_j/\max(\|e_j\|_2,10^{-12}),\quad m_g=|g|^{-1}\sum_{j\in g}u_j,\quad c=G^{-1}\sum_g m_g$",
       r"$x_g=[\,m_g;\ m_g\odot c;\ s_g;\ h_g\,],\quad z_g=a\odot\mathrm{clip}((x_g-\mu)/\sigma,-8,8)$",
       r"$P(g\mid q)=\frac{\exp(w^\top z_g)}{\sum_{\ell=1}^{G}\exp(w^\top z_\ell)},\qquad"
       r"\mathcal{L}=-\frac{1}{N}\sum_q\log P(y_q\mid q)+\frac{\lambda}{2}\|w\|_2^2$"]
    fig,ax=plt.subplots(figsize=(10.5,1.45))
    ax.axis("off")
    for i,eq in enumerate(equations):ax.text(.01,.86-i*.34,eq,fontsize=13,va="center")
    fig.savefig(FIG/"formulation.png",dpi=260,bbox_inches="tight",pad_inches=.05)
    plt.close(fig)
architecture();performance();ablation_plot();calibration_plot();equations()

# Shared content feeds both the PDF and the editable Markdown report.
pages=[]
def page(title):pages.append({"title":title,"blocks":[]})
def add(kind,**kwargs):pages[-1]["blocks"].append({"kind":kind,**kwargs})
def p(text):add("p",text=text)
def h(text):add("h",text=text)
def note(text):add("note",text=text)
def table(headers,rows,widths=None,caption=None):
    add("table",headers=headers,rows=rows,widths=widths,caption=caption)
def figure(name,caption,width=505,height=None):
    add("figure",name=name,caption=caption,width=width,height=height)
def code(text,language="bash"):add("code",text=text.strip("\n"),language=language)

page("1. Algorithm and research question")
add("title",text="Compact Answer-Group Ranking\nfor Math Solution Selection")
add("author",text="li shanghang  |  Student ID: 3036705979  |  DASE7506 MP2, Fall 2026")
note("Final public-bank selection accuracy: 393/467 = 84.15418%. "
     "A 2,112-input linear scorer uses 29,766 bytes of inference assets.")
h("Task and motivation")
p("Given a mathematics question, four fixed candidate solutions and their 1,024-dimensional representations, "
  "the task is to return one candidate index. Any candidate with the correct final numeric answer earns credit; "
  "the evaluator does not certify intermediate reasoning. Each supplied question has at least one correct candidate [1].")
p("The central question is whether inexpensive, deterministic evidence from the candidate texts can improve "
  "answer selection beyond majority voting and embedding centrality. The design combines agreement, semantic "
  "representation and local text/math evidence in a compact scorer. It uses the supplied representations rather "
  "than generating new solutions or loading an additional language model.")
figure("architecture","Figure 1. Inference groups equal parsed answers and scores shared group features. "
       "The dashed connection carries train-fitted weights and statistics; references are used only in training.",
       height=225)
h("Why rank answer groups?")
p("Several candidates can express the same correct value. Treating them as an answer group aligns the prediction "
  "unit with the official metric and represents agreement explicitly. A single valid answer group can be selected "
  "directly because of the bank guarantee. Otherwise, the system ranks all distinct parsed values and returns the "
  "lowest candidate index in the winning group. Unparseable candidates do not create an answer group.")
p("The final model adds 24 group columns derived from 12 lexical/numeric signals. A larger 80-column extension "
  "matched its validation count but had lower training OOF accuracy. This supports retaining the smaller system, "
  "while the results remain evidence about this course bank rather than a proof of general mathematical reasoning.")

page("2. Representation and training objective")
h("Canonical groups and embedding context")
p("The unchanged official parser uses the last #### suffix, or the last supported flat boxed expression, "
  "and canonicalizes the first extracted number with Fraction. Thus equivalent decimals and fractions share a group. "
  "For G answer groups, normalize each supplied embedding, average within a group, and form a context that weights "
  "each answer group equally. The equations below define the final representation and objective.")
figure("formulation","Here s_g has 40 existing scalar slots, h_g has 24 new columns, and a disables constant dimensions. "
       "The group softmax is used for training; inference selects the maximum score.",height=75)
table(["Block","Columns","Construction"],[
 ["Embedding mean and interaction","2,048","Group mean m_g and elementwise product m_g x c."],
 ["Group statistics","16","Vote shares/margins, entropy, within/cross similarity and embedding norms."],
 ["Arithmetic slots","8","Six enabled: step count, invalid fraction, final-step/trace matches, and any-match maxima; two slots remain zero."],
 ["Earlier text extension","16","Mean/max of eight signals: length, numeric diversity, content overlap, equation coverage and trace/number coverage."],
 ["Final lexical/numeric extension","24","Mean/max of the twelve candidate signals summarized below."]],[123,55,327],
 caption="Table 1. Final input layout: 2,048 + 16 + 8 + 16 + 24 = 2,112 columns; 2,110 are active.")
table(["New candidate family","Signals (12 total)"],[
 ["Content overlap (4)","Target-sentence recall in body, last three nonempty body lines and last line; question/last-line Jaccard."],
 ["Units (1)","Recall of fixed target-unit words in the last body line."],
 ["Format/counts (3)","log1p of body line count, raw equals count and final-answer mentions in body."],
 ["Numeric priors/diversity (4)","Nonnegative-integer flag; bounded signed log answer; answer/question-max log difference; fraction of novel body numbers."]],[128,377],
 caption="Table 2. The final-answer marker is excluded from the body. Fixed regexes, stop words and a small unit vocabulary are used.")
h("Local arithmetic is evidence, not proof")
p("A bounded recursive-descent parser checks numeric expressions, including fractions, percentages and implicit "
  "multiplication, with exact rational arithmetic and tolerance strictly below 1/1000. Unsupported variables, truncated "
  "runs and division by zero are skipped. This feature tolerance does not alter the official exact final-answer grading. "
  "Correct calculations can still encode the wrong interpretation.")
h("Optimization and determinism")
p("The final linear model uses group cross-entropy and lambda = 0.03, optimized by CPU L-BFGS-B (maximum 400 iterations). "
  "Each training question contributes total standardization weight one, split equally over its valid groups; scaling "
  "and constant-column detection are fitted inside each training fold, followed by clipping to [-8,8]. "
  "Training optimization uses float64; exported weights/statistics use float32 and scoring uses float64. "
  "Members are aggregated in stable text/embedding-byte order, with deterministic score ties.")

page("3. Experimental protocol and main results")
h("Data, supervision and selection")
table(["Split","Questions","Single answer group","Conflicting groups"],[
 ["Train","4,755","2,641","2,114"],["Validation","798","443","355"],["Test","467","0","467"]],[100,100,155,150],
 caption="Table 3. Counts from the unchanged course bank. Fixed embeddings have shape [N,4,1024].")
p("Only training inputs and references fit weights and standardization. The 2,114 conflict questions provide "
  "nonzero group-ranking supervision; the other training questions need no learned decision. Five folds keep all "
  "groups of a question together and stratify by the answer-vote pattern, using seed 7506. "
  "For each final feature variant, L2 in {0.01,0.03,0.10} is selected by train OOF accuracy, then NLL and stronger L2.")
p("The final round required all selected-fold and final optimizers to converge and OOF correct >= 1,745/2,114, "
  "the incumbent result. Among eligible models, validation accuracy was primary, followed by OOF accuracy, "
  "OOF NLL and parameter count. The chosen lexical checkpoint was frozen before its test evaluation.")
rows=[]
for k,r in D["models"].items():
    rows.append([r["name"],f"{r['oof']['accuracy']:.2f}%" if r["oof"] else "N/A",
       f"{r['val_correct']}/798 ({r['val_accuracy']:.2f}%)",
       f"{r['test']['correct']}/467 ({r['test']['accuracy']:.2f}%)"])
table(["Method","Train OOF","Validation","Test"],rows,[137,65,150,153],
 caption="Table 4. Archived results [3]. OOF covers training conflicts; validation/test are complete banks. N/A denotes training-free methods.")
figure("performance","Figure 2. Final-answer selection accuracy. The final method gains 70 test questions over voting, "
       "18 over the original linear model, and 5 over the previous expanded residual model.",height=165)
p("The final score is 84.1541755888651%, the official scorer's accuracy field. Validation is 751/798 (94.11%); "
  "on its conflict subset it is 308/355 (86.76%). Because the complete test contains only conflicts, "
  "the full-validation and test percentages describe different difficulty mixtures.")
note("Historical public test scores were known during subsequent development. Baseline and expanded-linear test "
     "scores in Table 4 were added after the final model was frozen, for reporting only. This is not a completely "
     "unseen-test experiment; the full development history is discussed in Section 8.")

page("4. Feature ablations and unsuccessful ideas")
h("How much handcrafted expansion is useful?")
arows=[]
for label,key in [("Lexical","lexical"),("Arithmetic trace","trace"),("Combined","combined"),("Full contextual","contextual")]:
    r=D["ablations"][key]
    arows.append([label,str(r["dim"]),str(r["l2"]),f"{r['oof']['correct']}/2114",
      f"{r['validation']['correct']}/798",f"{r['conflict']['correct']}/355",f"{r['corrected']}/{r['regressed']}"])
table(["Variant","Dim.","L2","OOF","Val.","Conflict","Fix/loss"],arows,[89,41,38,88,70,80,99],
 caption="Table 5. Fix/loss counts are paired against the previous expanded residual model. Only lexical was taken to test in this round.")
p("The lexical block increases validation from 734 to 751 correct. Combined features achieve the highest OOF "
  "count (1,763), but only 748 validation correct. The full contextual variant appends group-relative Z scores "
  "and vote interactions to all 48 aggregate columns; it ties lexical on validation but has lower OOF (1,756 vs 1,761). "
  "The predeclared tie-break therefore selects lexical.")
figure("l2_ablation","Figure 3. The fixed three-value L2 search uses train OOF only. More columns or weaker regularization "
       "do not produce a monotonic gain.",height=165)
h("Earlier experiments that did not justify adoption")
p("Expanded L2 search and block-specific regularization did not beat the six-math-feature model's 733/798 validation "
  "result. Two vote/math interactions also tied validation while reducing OOF accuracy. These outcomes were retained "
  "in the experiment logs rather than presented as improvements.")
p("A weakly regularized 16-unit residual head reached 737/798 validation but only 79.28% OOF, hit the iteration cap, "
  "and scored 376/467 (80.51%) on its frozen test, below the arithmetic model's 384/467. This was an unsuccessful "
  "earlier stage. Separate follow-ups predeclared convergence/OOF eligibility before their own training; this is a "
  "change in the research protocol, not a rule retrospectively applied to that earlier result.")
p("With stronger regularization, the previous eight-unit expanded residual model scored 388/467. Its hidden/output "
  "norms were approximately 0.003 and its validation choices matched expanded linear, so the record does not isolate "
  "a nonlinear benefit. The final system consequently keeps a linear scorer.")
note("These are practical pipeline comparisons, with L2 selected separately for each feature set. They do not "
     "establish causal contributions of individual signals. In particular, the lexical improvement also accompanies "
     "a change from the previous linear penalty 0.01 to 0.03.")

page("5. Validation analysis and uncertainty")
h("Paired improvements and remaining difficulty")
prows=[]
for key in ("vote","linear","residual"):
    r=D["paired_validation"][key]
    prows.append([D["models"][key]["name"],str(r["corrected"]),str(r["regressed"]),
       f"+{r['difference_pp']:.2f}",f"[{r['bootstrap_95ci_pp'][0]:.2f}, {r['bootstrap_95ci_pp'][1]:.2f}]"])
table(["Compared with","Fixed","Lost","Gain (pp)","95% bootstrap (pp)"],prows,[137,50,50,95,173],
 caption="Table 6. Paired full-validation differences; 2,000 question bootstrap resamples, seed 7506. Intervals are descriptive and unadjusted for model selection.")
p("Relative to the previous residual model, 21 questions are corrected and four are regressed: net +17. "
  "The gain interval [0.877,3.383] percentage points excludes zero under this descriptive bootstrap, but validation "
  "was also used to select the model. It is therefore not independent confirmation of a generalization effect.")
table(["Distinct groups","Questions","Original linear","Previous residual","Final lexical"],[
 [str(r["groups"]),str(r["n"]),f"{r['linear_correct']}/{r['n']}",
  f"{r['previous_correct']}/{r['n']}",f"{r['final_correct']}/{r['n']} ({r['final_accuracy']:.2f}%)"]
 for r in D["strata"]],[99,72,101,111,122],
 caption="Table 7. Validation by answer disagreement. All 443 single-group cases are automatically correct under the bank guarantee.")
p("The 47 remaining validation errors are all in conflicting groups. Accuracy falls from 92.07% with two groups "
  "to 77.78% with three and 75.00% with four. The four-group subset has only 20 questions; the observed ordering "
  "is suggestive rather than a stable estimate for every future bank.")
figure("validation_calibration","Figure 4. Post-freeze descriptive calibration on 355 validation conflicts only. "
       "Bins are [0,.4), [.4,.6), [.6,.8), [.8,.9), [.9,1]; low-count bins are noisy.",height=176)
p("Group softmax confidence is informative but imperfect. The five-bin expected calibration error, sum_b (n_b/355)|accuracy_b-confidence_b|, is 2.95 percentage "
  "points. Of 177 conflict questions assigned probability at least 0.9, six are wrong (96.61% accuracy). "
  "The lowest bin has only two questions. No probability calibrator, rejection threshold or new test-error rule "
  "was fitted during this report analysis.")

page("6. What the model learns, and where it fails")
h("A useful numeric prior is not mathematical understanding")
p("Post-freeze inspection finds 4,752 of 4,755 training references are nonnegative integers, three are negative integers, and none are fractions; "
  "all 798 validation references are nonnegative integers. This gives the answer-shape feature a strong empirical "
  "basis, but it is a distributional prior rather than a stated guarantee for arbitrary audit inputs.")
coeff={r["name"]:r["weight"] for r in D["largest_new_coefficients"]}
table(["New standardized coefficient","Weight"],[
 ["Mean nonnegative-integer flag",f"{coeff['mean_answer_nonnegative_integer']:+.3f}"],
 ["Max nonnegative-integer flag",f"{coeff['max_answer_nonnegative_integer']:+.3f}"],
 ["Mean novel-body-number fraction",f"{coeff['mean_novel_body_number_fraction']:+.3f}"],
 ["Max novel-body-number fraction",f"{coeff['max_novel_body_number_fraction']:+.3f}"],
 ["Mean target/last-line recall",f"{coeff['mean_focus_lastline_recall']:+.3f}"]],[385,120],
 caption="Table 8. Selected large coefficients among the 24 new columns, not causal feature importances.")
p("Mean and max answer-only signals are identical within a canonical answer group, creating redundancy in the frozen "
  "layout. Correlated features, this duplication and separate regularization choices make coefficient magnitude "
  "an unreliable measure of isolated importance. The negative target-recall coefficient also cautions against "
  "assuming that more literal word overlap is always helpful.")
h("Three illustrative validation cases")
p("The following cases were inspected after freezing for explanation only. They are zero-based validation rows; "
  "the prediction method never reads their IDs. Prompt summaries below are paraphrases.")
table(["Case","Reference / selected result","Observation"],[
 ["38: apples and bananas","Reference 3; previous 4.82; final 3 (p=0.803).",
  "The purchases imply 2x+y=7 and x+y=5. A wrong trace instead uses 12x+y=7. "
  "The final choice is correct, but neither this case nor numeric preference proves the scorer solved the equations."],
 ["412: garden rows","Reference 150; previous 150; final 160 (p=0.480).",
  "The chosen trace reads 'first 3 rows' as '1st and 3rd rows', then correctly computes "
  "20-(2+2)=16 and 16x10=160. Valid arithmetic supports a mistaken interpretation; the correct calculation is "
  "(20-3-2)x10=150."],
 ["175: relatives' ages","Reference 14; previous and final 2 (p=0.914).",
  "Three traces reverse the relation that the sister is twice Lexie's age. With Lexie 8, the brother is 2 "
  "and the sister 16, so the difference is 14. Agreement among three candidates reinforces the same semantic error."]
 ],[95,135,275],caption="Table 9. One corrected case, one regression and one high-confidence persistent error.")
p("The failure pattern is consistent with the design's limits: local numeric checks do not verify the meaning "
  "of quantities, modifiers or relationships. Future validation-only work could test unit/entity tracking, written "
  "number normalization and controlled removal of answer-shape priors. These remain proposals, not completed results.")

page("7. Resources and exact reproduction")
h("Measured cost and implementation checks")
table(["Item","Recorded final result"],[
 ["Local CPU / inference environment","Intel Core Ultra 9 275HX; Python 3.12.3; NumPy 1.26.4."],
 ["Training environment","CPU; SciPy 1.11.4. Final four-variant search about 91 s; total experiment 97.8 s."],
 ["Complete fresh prediction + independent scoring","1.166 s, including startup/imports, hashes, loading and preprocessing."],
 ["Threads / additional inference assets","Observed total peak 2; documented child limit 3. Three files total 29,766 bytes (29.07 KiB)."],
 ["Stored arrays","Float32 weight/mean/scale and boolean active mask; 27,456 array bytes. NPZ members are uncompressed."],
 ["Reference-machine eligibility","Not verified on the required Xeon Platinum 8457C; local measurements do not establish eligibility."]],[185,320],
 caption="Table 10. Resource claims concern the final local run, not a controlled cross-stage speed comparison. File-cache state was not controlled.")
p("31 unit tests passed. Real validation/test replays, candidate permutations and different batch sizes preserved "
  "selected answers; all 822 conflict feature matrices were bitwise identical after candidate permutation. "
  "A guard blocking data reopens, cached experiment files, network access and subprocesses still allowed inference. "
  "Fixed-configuration retraining reproduced weights, standardization and OOF arrays bitwise in the recorded environment.")
h("Predict and score without retraining")
p("Run from the repository root with Python 3.12. Use the solution-specific pins; the original course requirements "
  "file specifies a different NumPy version. The exact expected accuracy is 84.1541755888651.")
code(r"""cd code
python3 -m venv solution/.venv-inference
solution/.venv-inference/bin/python -m pip install \
  -r solution/residual_ranker/requirements-inference.txt
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
export MKL_NUM_THREADS=3 BLIS_NUM_THREADS=3
solution/.venv-inference/bin/python predict.py \
  --method solution/residual_ranker/extended_submission.py \
  --checkpoint solution/runs/handcrafted_extended_v2/models/lexical/checkpoint \
  --split test --output solution/runs/report_reproduction/predictions.json
solution/.venv-inference/bin/python score_predictions.py --split test \
  --predictions solution/runs/report_reproduction/predictions.json \
  --output solution/runs/report_reproduction/score.json""")
h("Retrain the fixed configuration and run tests")
code(r"""python3 -m venv solution/.venv-train
solution/.venv-train/bin/python -m pip install -r solution/requirements-train.txt
solution/.venv-train/bin/python -m solution.residual_ranker.extended_train \
  --schemas lexical --l2 0.03 --output solution/runs/report_retrain
solution/.venv-train/bin/python -m unittest discover -s solution -p 'test_*.py'""")
p("The checkpoint must include model.npz, config.json and asset_manifest.json. The final freeze ID and full model/config "
  "hashes are listed in report/results.json and the original freeze.json. The frozen model version is "
  "answer-group-extended-linear-v1. New runs are Git-ignored, so matching assets must be included in the final release.")

code("Model SHA256: " + D["checkpoint_hashes"]["model.npz"] + "\n" +
     "Config SHA256: " + D["checkpoint_hashes"]["config.json"] + "\n" +
     "Code freeze: " + D["freeze_id"],language="text")

page("8. Interpretation, integrity and conclusion")
h("Experimental history and limits of the claim")
p("Historical test correct counts (out of 467) were: original linear 375; arithmetic 384; "
  "optimization replay 384; weak residual 376; expanded residual 388; final lexical 393. "
  "Baseline/expanded-linear test comparisons were reporting runs after the final freeze. "
  "Failed results and prior decisions remain in their original directories.")
p("The final-round grid and eligibility rules were recorded before its training, and current selection uses "
  "train OOF and validation. Nevertheless, earlier public test scores were known while later development continued. "
  "A per-round freeze does not erase that history or establish complete test independence. Repeated validation "
  "selection also introduces optimism. These results should be treated as public-bank development evidence, with "
  "generalization and formal reference-machine eligibility still requiring independent audit.")
h("Alignment with the assignment")
table(["Requirement","Evidence / boundary"],[
 ["Training supervision and fixed benchmark","Only train labels fit parameters; official package hashes match. "
  "Predict reads no references; the scorer reads test labels."],
 ["Offline selection and new inputs","One supplied candidate index; no external data/weights, LLM calls, ID lookup or cached choices. "
  "New-bank, ID and synthetic-input checks passed."],
 ["Limits and reproduction","Local limits passed; four-thread cap required; reference CPU test pending. "
  "Matching public code/checkpoint links are still required."],
 ["Analysis and disclosure","Baseline/ablation/failure/resource evidence is retained. "
  "Substantive AI/reuse assistance is disclosed below."]],[155,350],
 caption="Table 11. Implementation checks support these statements; they do not adjudicate the full historical development process.")
h("AI assistance, reuse and data attribution")
p("Codex assisted with design, feature/model code, tests, experiments, analysis, "
  "documentation and drafting this report. Python generated the figures and PDF. "
  "The submitter remains responsible for understanding, verification and submission. "
  "The project reuses course tools/data/representations and its earlier group-ranker functions.")
p("Questions/references derive from the course-provided GSM8K derivative under its retained MIT license. "
  "Candidate traces and fixed representations were supplied using Qwen3-0.6B, revision "
  "c1899de289a04d12100db370d81485cdf75e47ca; no additional Qwen weights were downloaded. "
  "The data LICENSE.txt and NOTICE.txt must accompany redistribution [1,2].")
h("Conclusion")
p("A compact linear answer-group ranker with deterministic lexical/numeric evidence reaches 84.15418% on the full "
  "course test bank, improving on majority voting and the original linear ranker. The useful evidence combines "
  "agreement, embeddings and dataset priors; larger expansions and nonlinear capacity did not consistently help. "
  "The principal unresolved limitations are semantic reasoning errors, public-bank selection bias and unverified "
  "reference-machine performance.")
h("References and evidence")
p("[1] DASE7506 MP2 course guide and code README, Fall 2026: guide/GUIDE.md; code/README.md. "
  "Course bank protocol: dase7506-mp2-open-hard-verifier-v5.")
p("[2] Supplied data provenance: code/data/manifest.json, NOTICE.txt and LICENSE.txt; "
  "GSM8K: https://github.com/openai/grade-school-math; Qwen source: https://huggingface.co/Qwen/Qwen3-0.6B.")
p("[3] Evaluation evidence: code/solution/runs/ (the run directories identified in Sections 3-4). "
  ""
  "Report-derived validation diagnostics and hashes: report/results.json.")

# Typesetting.
FONTDIR=Path("/usr/share/fonts/truetype/dejavu")
for name,file in [("Body","DejaVuSans.ttf"),("Body-Bold","DejaVuSans-Bold.ttf"),
                  ("Body-Oblique","DejaVuSans-Oblique.ttf"),("Mono","DejaVuSansMono.ttf")]:
    pdfmetrics.registerFont(TTFont(name,str(FONTDIR/file)))
pdfmetrics.registerFontFamily("Body",normal="Body",bold="Body-Bold",italic="Body-Oblique")
styles={
 "body":ParagraphStyle("body",fontName="Body",fontSize=9.1,leading=12.7,spaceAfter=7,textColor=colors.HexColor(NAVY)),
 "section":ParagraphStyle("section",fontName="Body-Bold",fontSize=16.1,leading=20,spaceAfter=12,textColor=colors.HexColor(NAVY)),
 "sub":ParagraphStyle("sub",fontName="Body-Bold",fontSize=10.8,leading=14,spaceBefore=7,spaceAfter=6,textColor=colors.HexColor(TEAL)),
 "title":ParagraphStyle("title",fontName="Body-Bold",fontSize=23,leading=28,spaceAfter=10,textColor=colors.HexColor(NAVY)),
 "author":ParagraphStyle("author",fontName="Body",fontSize=9.2,leading=13,spaceAfter=12,textColor=colors.HexColor(GREY)),
 "caption":ParagraphStyle("caption",fontName="Body",fontSize=7.6,leading=10.1,spaceAfter=7,textColor=colors.HexColor("#586c79")),
 "cell":ParagraphStyle("cell",fontName="Body",fontSize=8.0,leading=10.5,textColor=colors.HexColor(NAVY)),
 "headcell":ParagraphStyle("headcell",fontName="Body-Bold",fontSize=8.0,leading=10.5,textColor=colors.white),
 "note":ParagraphStyle("note",fontName="Body",fontSize=8.7,leading=12.3,textColor=colors.HexColor(NAVY),spaceAfter=9),
 "code":ParagraphStyle("code",fontName="Mono",fontSize=6.7,leading=8.9,spaceAfter=8),
}
def para(text,style="body"):
    return Paragraph(escape(text).replace("\n","<br/>"),styles[style])
W,H=A4
BODY_W=W-88
def pdf_blocks(page):
    out=[para(page["title"],"section")]
    for b in page["blocks"]:
        k=b["kind"]
        if k in ("p","h","title","author"):
            style={"p":"body","h":"sub","title":"title","author":"author"}[k]
            out.append(para(b["text"],style))
        elif k=="note":
            t=Table([[para(b["text"],"note")]],colWidths=[BODY_W])
            t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),colors.HexColor(LIGHT)),
                ("BOX",(0,0),(-1,-1),.5,colors.HexColor("#d3e2e7")),
                ("LEFTPADDING",(0,0),(-1,-1),9),("RIGHTPADDING",(0,0),(-1,-1),9),
                ("TOPPADDING",(0,0),(-1,-1),8),("BOTTOMPADDING",(0,0),(-1,-1),1)]))
            out.extend([t,Spacer(1,7)])
        elif k=="table":
            cells=[[para(x,"headcell") for x in b["headers"]]]
            cells += [[para(str(x),"cell") for x in row] for row in b["rows"]]
            widths=b["widths"] or [BODY_W/len(cells[0])]*len(cells[0])
            widths=[float(w)*BODY_W/sum(widths) for w in widths]
            t=Table(cells,colWidths=widths,repeatRows=1,hAlign="LEFT")
            t.setStyle(TableStyle([
               ("BACKGROUND",(0,0),(-1,0),colors.HexColor(NAVY)),
               ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f1f5f7")]),
               ("VALIGN",(0,0),(-1,-1),"TOP"),
               ("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6),
               ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
               ("LINEBELOW",(0,-1),(-1,-1),.4,colors.HexColor("#b8c9d1")),
            ]))
            out.append(t)
            if b.get("caption"):out.append(para(b["caption"],"caption"))
            else:out.append(Spacer(1,7))
        elif k=="figure":
            path=FIG/(b["name"]+".png")
            im=Image(str(path))
            width=min(b["width"],BODY_W)
            height=b["height"] or im.imageHeight/im.imageWidth*width
            im.drawWidth=width;im.drawHeight=height;im.hAlign="CENTER"
            out.extend([im,para(b["caption"],"caption")])
        elif k=="code":
            # Preformatted blocks are explicitly line-broken and checked for overflow.
            for line in b["text"].splitlines():
                assert pdfmetrics.stringWidth(line,"Mono",6.7)<BODY_W-18,line
            pre=Preformatted(b["text"],styles["code"])
            t=Table([[pre]],colWidths=[BODY_W])
            t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),colors.HexColor("#f2f5f6")),
               ("BOX",(0,0),(-1,-1),.4,colors.HexColor("#d3e0e5")),
               ("LEFTPADDING",(0,0),(-1,-1),8),("RIGHTPADDING",(0,0),(-1,-1),8),
               ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),3)]))
            out.extend([t,Spacer(1,6)])
    return out
def header_footer(c,doc):
    c.saveState()
    c.setStrokeColor(colors.HexColor("#b5c9d2"));c.setLineWidth(.55)
    c.line(44,H-32,W-44,H-32)
    c.setFillColor(colors.HexColor("#526b7b"));c.setFont("Body",7.1)
    c.drawString(44,H-23,"DASE7506 MP2  |  Compact Answer-Group Ranking")
    c.drawRightString(W-44,H-23,"li shanghang  |  3036705979")
    c.line(44,32,W-44,32)
    c.drawString(44,21,"Final frozen model: lexical / linear L2 = 0.03")
    c.drawRightString(W-44,21,f"{doc.page} / 8")
    c.restoreState()
out=HERE/"MP2_Final_Report.pdf"
story=[]
layout=[]
for i,pg in enumerate(pages):
    flows=pdf_blocks(pg)
    total=0
    for f in flows:
        _,height=f.wrap(BODY_W,10000)
        total+=height+f.getSpaceBefore()+f.getSpaceAfter()
    layout.append({"section":pg["title"],"estimated_height_pt":round(total,1)})
    story.extend(flows)
    if i<len(pages)-1:story.append(PageBreak())
doc=SimpleDocTemplate(str(out),pagesize=A4,leftMargin=44,rightMargin=44,
    topMargin=47,bottomMargin=43,title="Compact Answer-Group Ranking for Math Solution Selection",
    author="li shanghang (3036705979)",subject="DASE7506 MP2 final report",
    pageCompression=1)
doc.build(story,onFirstPage=header_footer,onLaterPages=header_footer)
reader=PdfReader(str(out))
assert len(reader.pages)==len(pages)==8,f"Unexpected pagination: {len(reader.pages)}"
md=["# Compact Answer-Group Ranking for Math Solution Selection",
    "", "**Author:** li shanghang  ", "**Student ID:** 3036705979  ",
    "**Course:** DASE7506 MP2, Fall 2026", ""]
for pg in pages:
    md+=["## "+pg["title"],""]
    for b in pg["blocks"]:
        k=b["kind"]
        if k=="title":continue
        if k=="author":continue
        if k=="h":md+=["### "+b["text"],""]
        elif k=="p":md += [b["text"],""]
        elif k=="note":md += ["> "+b["text"],""]
        elif k=="table":
            md+=["| "+" | ".join(b["headers"])+" |","|"+"|".join(["---"]*len(b["headers"]))+"|"]
            md += ["| "+" | ".join(str(x).replace("|","/") for x in row)+" |" for row in b["rows"]]
            md += ["",b.get("caption",""),""]
        elif k=="figure":md += [f"![{b['caption']}](figures/{b['name']}.png)","",b["caption"],""]
        elif k=="code":md += ["~~~"+b.get("language","bash"),b["text"],"~~~",""]
(HERE/"REPORT.md").write_text("\n".join(md)+"\n")
(HERE/"layout.json").write_text(json.dumps({"pdf_pages":len(reader.pages),"sections":layout},indent=2)+"\n")
print(json.dumps({"pdf":str(out),"pages":len(reader.pages),"layout":layout},indent=2))
