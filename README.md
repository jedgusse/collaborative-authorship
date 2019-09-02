![front cover of thesis](https://github.com/jedgusse/collaborative-authorship/blob/master/front-page.png)

# De Gussem, Jeroen. 'Collaborative Authorship in Twelfth-Century Latin Literature. A Stylometric Approach to Gender, Synergy and Authority'. Ghent University and University of Antwerp, 2019.

This GitHub repository contains the text data and accompanying Python code used in the PhD thesis 'Collaborative Authorship in Twelfth-Century Latin Literature. A Stylometric Approach to Gender, Synergy and Authority', to be defended at Ghent University and the University of Antwerp in October 2019.
The rationale behind this page is to allow for the replication of experiments conducted throughout this thesis's chapters.

# Data

Aside from a number of exceptions, the pool of texts used for the experiments carried out in the thesis roughly derive from two main databases. 

## Brepols Library of Latin Texts

The first is the Brepols Library of Latin Texts (LLT).
The LLT first and foremost contains all the editions from Brepols's own *Corpus Christianorum*, in addition to several external critical editions that comply to modern critical standards.
Aside from being able to rely on the texts that had previously been collected in LLT, I have been fortunate to collaborate with Brepols on the further digitization of texts which had thusfar not been available in electronic edition, such as Ewald Könsgen's edition of the *Epistolae duorum amantium* (1974), Suger of Saint-Denis's collected oeuvre as edited by Françoise Gasparri (1996–2001), or the respective works by Elisabeth and Ekbert of Schönau in the edition by Ferdinand W. Roth (1884).

## *Patrologia Latina*

For the remaining texts, I chiefly relied on the digitized version of the *Patrologia Latina* (*PL*), which has become electronically available since 1993, and has remained one of the most sizeable Latin corpora online (±113 million words).
The *PL* is a corpus containing texts of Latin ecclesiastical writers in 221 volumes ranging a time span of ten centuries, from Late Antiquity to the High Middle Ages (Tertullian c. 200 to Pope Innocent III c. 1216). 
The *PL* was first published in two series halfway the nineteenth century by the Parisian priest and theologian Jacques-Paul Migne (1800-1875), who mainly drew on seventeenth and eighteenth-century prints to compile the patristic heritage.
Although Migne's conservation of a great number of patristic texts which would otherwise have gone lost unmistakably stands as a major contribution to medieval studies, his critical attitudes have been deficient, which has from time to time caused concerns over the corpus's overall reliability in matters of authenticity and ascription. 
However, the *PL*'s accessibility and encyclopedic quality render it a useful tool for scholars, and some texts can be consulted there and only there.

## Preprocessing and Formatting

Medieval texts have different orthographical appearances, and editors of texts in the LLT or those in the PL apply different rules and practices in transcribing texts and of handling and displaying the various witnesses. 
It stands beyond question that such differences constitute a poor ground upon which to automatically compare texts on a large scale, which is why they need to be addressed prior to proceeding to stylistic analysis. 
In natural language processing (NLP), this task commonly falls under 'preprocessing,' which entails minor interventions in the text such as the deletion of irrelevant textual material and the normalization of divergent orthographical forms.
In a nutshell, preprocessing Latin texts enables us to automatically align orthographical differences between such variant appearances of lexical items such as the pairs *racio* and *ratio*, or *aliquandiu* and *aliquamdiu*.

The original texts have been slightly camouflaged so as to respect the copyright laws protecting the editions. 
Only function words —which are highly successful for distinguishing writing styles— were retained in their original position and form.
All the remaining, content-loaded words, were substituted by 'dummy words', rendering the text illegible. 
This means that some experiments in this thesis, those which relied on most-frequent content words in addition to function words, will not be replicable by relying solely on the text data as available on GitHub. 
To replicate these experiments, one may request access to the electronic versions of the editions referred to by contacting [Brepols Library of Latin Texts](http://clt.brepolis.net/llta/).

## Included authors

|	Author 		| 	Texts 	 | 
|------------|-------------| 
| Alan of Lille (c. 1128–c. 1203) <br> *Alanus de Insulis* | *Anticlaudianus* |
|—|*Contra haereticos* |
|—|*De arte praedicatoria* |
|—|*De planctu naturae* |
|—|*Elucidatio in Cantica canticorum* |
|—|*Sermones* |
|—|*Summa "Quoniam homines"* |
| Anselm of Canterbury (1033–1109) <br> *Anselmus Cantuariensis* | *Cur deus homo* |
|—|*Monologion* |
|—|*Proslogion* |
| Anselm of Laon († 1117) <br> *Anselmus Laudunensis* | *Enarrationes in Apocalypsin* |
|—|*Enarrationes in Cantica canticorum* |
| Bruno of Cologne (1030–1101) <br> *Bruno Carthusianorum* | *Expositio in epistolas Pauli* |
|—|*Expositio in Psalmos* |
| Gerhoh of Reichersberg (1093–1169) <br> *Gerhohus Reicherspergensis* | *Commentarius aureus in Psalmos et cantica ferialia* |
|—|*De aedificio dei* |
|—|*Epistolae Gerhohi* |
| Gilbert of Poitiers (1075–1154) <br> *Gislebertus Porretanus* | *Expositio in Boethii librum De bonorum hedbomade* |
|—| *Liber de sex principiis* |
| Guibert de Nogent (c. 1053–1125) <br> *Guibertus de Novigento* | *Carmen Erumnarum et dolorum* |
|—| *Contra iudaizantem et Iudaeos* |
|—| *De bucella Iudae data* |
|—| *De sanctis et eorum pigneribus* |
|—| *De vita sua sive Monodiae* |
|—| *Historia quae inscribitur Dei gesta per Francos* |
|—| *In Zachariam* |
|—| *Quo ordine sermo fieri debeat* |
| Hildebert of Lavardin (c. 1055–1133) <br> *Hildebertus Lavardinensis* | *Epistolae de Paschali papa* |
|—| *Vita Mariae Aegyptiacae* |
| Honorius of Regensburg (c. 1080–c. 1154) <br> *Honorius Augustodunensis* | *De apostatis* |
|—| *De offendiculo*|
|—| *Elucidarium*|
|—| *Expositio in Cantica canticorum*|
|—| *Gemma animae*|
|—| *Imago mundi*|
|—| *Imago mundi: Continuatio Weingartensis*|
|—| *Imago mundi (cum continuationibus VII) (Excerpta) Speculum ecclesiae*|
|—| *Summa gloria* |
|—| *Summa totius* |
| Hugh of Saint-Victor (c. 1096–1141) <br> *Hugo de Sancto Victore* | *De Sacramentis Christianae fidei* |
|—| *De septem donis spiritus sancti* |
|—| *De substantia dilectionis* |
|—| *De tribus rerum subsistentiis* |
|—| *De vanitate rerum mundanarum* |
|—| *De verbo dei* |
|—| *Dialogus de creatione mundi* |
|—| *In hierarchicam coelestem* |
|—| *Sententiae de divinitate* |
| Ivo of Chartres (1040–1115) <br> *Ivo Carnotensis* | *Decretum* |
|—| *Epistolae ad litem investiturarum spectantes* |
|—| *Panormia* |
| John of Salisbury (c. 1125–1180) <br> *Ioannes Saresberiensis* | *Epistolae* |
|—| *Metalogicon* |
|—| *Polycraticus* |
|—| *Vita Sancti Anselmi Cantuariensis* |
| Peter Damian (c. 1007–c.1073) <br> *Petrus Damianus* | *De divina omnipotentia* |
|—| *Liber Gomorrhianus* |
|—| *Vita sancti Romualdi* |
| Peter Lombard (c. 1096–1160) <br> *Petrus Lombardus* | *Collectanea in epistolas Pauli* |
|—| *Commentaria in Psalmos* |
|—| *Sententiae* |
| Peter of Celle (c. 1115–1183) <br> *Petrus Cellensis* | *De conscientia* |
|—| *Epistolae* |
|—| *Liber de panibus* |
|—| *Mystica et moralis expositio Mosaici tabernaculi Sermones* |
| Peter the Venerable (1092–1156) <br> *Petrus Venerabilis* | *Adversus Iudaeorum inveteratam duritiem* |
|—| *Adversus sectam Saracenorum* |
|—| *Carmina* |
|—| *Contra Petrobrusianos haereticos* |
|—| *De miraculis libri duo* |
|—| *Epistulae* |
| Rupert of Deutz (c. 1075–1130) <br> *Rupertus Tuitiensis* | *Anulus sive dialogus inter Christianum et Iudaeum* |
|—| *Carmina de calamitatibus ecclesiae Leodiensis Commentaria in Canticum Canticorum* |
|—| *Commentaria in duodecim prophetas minores* |
|—| *Commentaria in evangelium sancti Iohannis Commentarium in Apocalypsim Iohannis apostoli* |
|—| *De gloria et honore* |
|—| *De glorificatione Trinitatis et processione Spiritus sancti De incendio Tuitiensi* |
|—| *De meditatione mortis* |
|—| *De sancta trinitate et operibus eius* |
|—| *De victoria verbi Dei* |
|—| *[?] Epistula ad F.* |
|—| *Hymnus primus de sancto spiritu* |
|—| *Hymnus secundus sive Oratio ad sanctum spiritum Liber de divinis offciis* |
|—| *Liber de laesione virginitatis* |
|—| *[?] Officium de festo Sancti Augustini* |
|—| *Passio Eliphii Tullensis* |
|—| *Sermo de Pantaleone* |
|—| *[?] Vita Heriberti Coloniensis* |
| Walter of Châtillon (c. 1135–c. 1179) <br> *Gualterus de Castellione* | *Alexandreis* |
|—| *[?] De SS. Trinitate tractatus* |
|—| *Tractatus contra Iudaeos* |
| William of Conches (c. 1090–1154) <br> *Guillelmus de Conchis* | *Dragmaticon Philosophiae* |
|—| *Glosae super Boetium* |
| William of Saint-Thièrry (c. 1080–1148) <br> *Guilhelmus de Sancto Theodorico* | *Aenigma fidei* |
|—| *Brevis commentatio (in Cantici Canticorum priora dua capita)* |
|—| *De contemplando Deo* |
|—| *Epistola de erroribus Guillelmi de Conchis* |
|—| *De natura corporis et animae* |
|—| *De natura et dignitate amoris* |
|—| *De sacramento altaris* |
|—| *Disputatio adversus Abaelardum* |
|—| *[?] Disputatio altera adversus Abaelardum* |
|—| *Expositio altera super Cantica canticorum* |
|—| *Expositio in Epistolam ad Romanos* |
|—| *Orationes meditativae* |
|—| *Speculum fidei* |

# Code

# Acknowledgements

I would like to express my gratitude to my supervisors, prof. dr. Jeroen Deploige, prof. dr. Wim Verbaal and prof. dr. Mike Kestemont.
Furthermore, I want to thank the [Ghent University Special Research Fund (BOF)](https://www.ugent.be/en/research/funding/bof), the [Research Foundation Flanders (FWO)](https://www.fwo.be/en/), the [Henri Pirenne Institute for Medieval Studies at Ghent University](https://www.ugent.be/pirenne/en), the [CLiPS Computational Linguistics Group at the University of Antwerp](https://www.clips.uantwerpen.be/computational-linguistics?page=4), and the *Centre Traditio Litterarum Occidentalium* division for computer-assisted research into Latin language and literature housed in the [*Corpus Christianorum Library and Knowledge Centre*](https://www.corpuschristianorum.org/) of Brepols Publishers in Turnhout (Belgium).