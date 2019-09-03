[//]: # (cmd : git add . && git commit -m "`date`" && git push collaborative-authorship master)

![front cover of thesis](https://github.com/jedgusse/collaborative-authorship/blob/master/frontpage.png)

# Welcome

This GitHub repository contains the text data and accompanying Python code used in the PhD thesis Jeroen De Gussem, 'Collaborative Authorship in Twelfth-Century Latin Literature. A Stylometric Approach to Gender, Synergy and Authority,' to be defended at Ghent University and the University of Antwerp in October 2019.
The rationale behind this repository is to allow for the replication of experiments conducted throughout this thesis's chapters, by offering open-source code and text data.

# Code

## Dependencies

The open-source [Anaconda Distribution](https://www.anaconda.com/distribution/) should be downloaded, as should [Python 3](https://www.python.org/downloads/).

## How to run

*Under construction...*

# Data

The pool of texts subjected to quantitative analysis in the thesis roughly derive from two main databases. 

## Brepols Library of Latin Texts

The first is the Brepols Library of Latin Texts (LLT).
The LLT first and foremost contains all the editions from Brepols's own *Corpus Christianorum*, in addition to several external critical editions that comply to modern critical standards.
Aside from texts previously collected in LLT, I have been fortunate to collaborate with Brepols on the further digitization of texts which had thusfar not been available in electronic edition, such as Ewald Könsgen's edition of the *Epistolae duorum amantium* (1974), Suger of Saint-Denis's collected oeuvre as edited by Françoise Gasparri (1996–2001), or the respective works by Elisabeth and Ekbert of Schönau in the edition by Ferdinand W. Roth (1884).

## *Patrologia Latina*

For the remaining texts, I chiefly relied on the digitized version of the *Patrologia Latina* (*PL*), which has become electronically available since 1993, and has remained one of the most sizeable Latin corpora online (±113 million words).
The *PL* is a corpus containing texts of Latin ecclesiastical writers in 221 volumes ranging a time span of ten centuries, from Late Antiquity to the High Middle Ages (Tertullian c. 200 to Pope Innocent III c. 1216). 
The *PL* was first published in two series halfway the nineteenth century by the Parisian priest and theologian Jacques-Paul Migne (1800-1875), who mainly drew on seventeenth and eighteenth-century prints to compile the patristic heritage.

## Preprocessing and Formatting

Medieval texts have different orthographical appearances, and editors of texts in the LLT or those in the *PL* apply different rules and practices in transcribing texts and of handling and displaying the various witnesses. 
It stands beyond question that such differences constitute a poor ground upon which to automatically compare texts on a large scale.
In natural language processing (NLP), the task of aligning variant appearances of lexical items, such as the pairs *racio* and *ratio*, or *aliquandiu* and *aliquamdiu*, commonly falls under 'preprocessing,' which entails minor interventions in the text such as the deletion of irrelevant textual material and the normalization of divergent orthographical forms.

Examples are: 

* \<j\>'s vs. \<i\>'s
* \<ae\>'s vs. \<e\>'s, or \<oe\>'s vs. \<e\>'s
* \<v\>'s vs. \<u\>'s
* lenition: *racio* vs. *ratio* or *multociens* vs. *multotiens*
* strengthened aspiration or fortition, e.g. *michi* vs. *mihi* (other examples: *nichil* vs. *nihil*)
* progressive or regressive assimilation / dissimilation: *exsistere* vs. *existere*, *obf-* vs. *off-* (e.g. *obfuscare*), *abji-* vs. *abi-* (e.g. *abjiciendus est*), *adm-* vs. *amm-* (e.g. *ammonitio*, *ammiratio*), etc.
* *quandiu* vs. *quamdiu* (other examples are: *tanquam* vs. *tamquam*, *nunquam* vs. *numquam*,
-*cunque* vs. -*cumque*, ...), *tandiu* vs. *tamdiu* (and *quandiu* vs. *quamdiu*)
* *quatenus* vs. *quatinus*
* *imo* vs. *immo*
* *quoties* vs. *quotiens* (other examples: *totiens* vs. *toties*)
* *velut* vs. *uelud*

The original texts found in the data folder in this repository have been camouflaged so as to respect the copyright laws protecting the editions.
Only function words —which are highly successful for distinguishing writing styles— were retained in their original position and form.
All the remaining, content-loaded tokens were substituted by asterisks, rendering the text illegible. 
This means that some experiments in the thesis, those which relied on most-frequent content words in addition to function words, will not be replicable by relying solely on the text data as available on GitHub. 
To replicate these experiments as well, one may request access to the electronic versions of the editions referred to by contacting [Brepols Library of Latin Texts](http://clt.brepolis.net/llta/) or request access to the online *PL* through [Chadwyck](http://pld.chadwyck.co.uk/).

## Included Authors

Detailed references given in the thesis. 

|	Authors 		| 	Texts 	 | 
|------------|-------------| 
| Alan of Lille (c. 1128–c. 1203) <br> *Alanus de Insulis* | *Anticlaudianus* |
|━━| *Contra haereticos* |
|━━| *De arte praedicatoria* |
|━━| *De planctu naturae* |
|━━| *Elucidatio in Cantica canticorum* |
|━━| *Sermones* |
|━━| *Summa "Quoniam homines"* |
| Anselm of Canterbury (1033–1109) <br> *Anselmus Cantuariensis* | *Cur deus homo* |
|━━| *Monologion* |
|━━| *Proslogion* |
| Anselm of Laon († 1117) <br> *Anselmus Laudunensis* | *Enarrationes in Apocalypsin* |
|━━| *Enarrationes in Cantica canticorum* |
| Bernard of Clairuaux (1090–1153) <br> *Bernardus Claraeuallensis* | *Apologia ad Guillelmum abbatem* |
|━━| *De consideratione libri u tractatus* |
|━━| *Ep. de moribus et officio espiscoporum (ep. 42)* |
|━━| *Epistula de baptismo* |
|━━| *Epistula de erroribus Petri Abaelardi* |
|━━| *Epistulae nuper repertae* |
|━━| *Epistulae* |
|━━| *Epitaphium de sancto Malachia* |
|━━| *Homiliae super 'Missus est'* |
|━━| *Hymnus de sancto Malachia* |
|━━| *Liber ad milites Templi De laude nouae militiae* |
|━━| *Liber de diligendo Deo* |
|━━| *Liber de gradibus humilitatis et superbiae* |
|━━| *Liber de gratia et de libero arbitrio* |
|━━| *Liber de praecepto et dispensatione* |
|━━| *Officium de sancto Uictore* |
|━━| *Parabolae editae in An.S.O.Cist. et in Cîteaux* |
|━━| *Parabolae* |
|━━| *Prologus in antiphonarium* |
|━━| *Sententiae* |
|━━| *Sermo ad abbates* |
|━━| *Sermo de altitudine et bassitudine cordis* |
|━━| *Sermo de conuersione ad clericos (textus breuis)* |
|━━| *Sermo de conuersione ad clericos (textus longior)* |
|━━| *Sermo de misericordiis* |
|━━| *Sermo de sancto Malachia* |
|━━| *Sermo de septem donis Spiritus Sancti* |
|━━| *Sermo de uoluntate diuina* |
|━━| *Sermo in aduentu Domini* |
|━━| *Sermo in cena Domini* |
|━━| *Sermo in conuersione sancti Pauli ('Conuersus est')* |
|━━| *Sermo in conuersione sancti Pauli ('Merito quidem')* |
|━━| *Sermo in dom. inf. octauam assumptionis b. Mariae* |
|━━| *Sermo in dominica quarta post pentecosten* |
|━━| *Sermo in epiphania Domini* |
|━━| *Sermo in f. ss. Stephani, Iohannis et Innocentium* |
|━━| *Sermo in feria iu hebdomadae sanctae* |
|━━| *Sermo in festo sancti Martini* |
|━━| *Sermo in natali sancti Benedicti* |
|━━| *Sermo in natali sancti Clementis* |
|━━| *Sermo in natiuitate beatae Mariae Uirginis* |
|━━| *Sermo in natiuitate sancti Iohannis Baptistae* |
|━━| *Sermo in obitu domini Humberti* |
|━━| *Sermo in octaua epiphaniae Domini* |
|━━| *Sermo in rogationibus* |
|━━| *Sermo in transitu sancti Malachiae* |
|━━| *Sermo in uigilia sancti Andreae* |
|━━| *Sermo in uigilia sanctorum Petri et Pauli* |
|━━| *Sermones de diuersis* |
|━━| *Sermones in adnuntiatione dominica* |
|━━| *Sermones in aduentu Domini* |
|━━| *Sermones in ascensione Domini* |
|━━| *Sermones in assumptione beatae Mariae Uirginis* |
|━━| *Sermones in circumcisione Domini* |
|━━| *Sermones in dedicatione ecclesiae* |
|━━| *Sermones in die paschae* |
|━━| *Sermones in die pentecostes* |
|━━| *Sermones in dominica i nouembris* |
|━━| *Sermones in dominica i post octauam epiphaniae* |
|━━| *Sermones in dominica sexta post pentecosten* |
|━━| *Sermones in epiphania Domini* |
|━━| *Sermones in festiuitate omnium sanctorum* |
|━━| *Sermones in festo sancti Michaelis* |
|━━| *Sermones in festo sanctorum apostolorum Petri et Pauli* |
|━━| *Sermones in labore messis* |
|━━| *Sermones in natali sancti Andreae* |
|━━| *Sermones in natali sancti Uictoris* |
|━━| *Sermones in natiuitate Domini* |
|━━| *Sermones in octaua paschae* |
|━━| *Sermones in purificatione beatae Mariae Uirginis* |
|━━| *Sermones in quadragesima* |
|━━| *Sermones in ramis palmarum* |
|━━| *Sermones in septuagesima* |
|━━| *Sermones in uigilia natiuitatis Domini* |
|━━| *Sermones super Cantica Canticorum* |
|━━| *Sermones super psalmum 'Qui habitat'* |
|━━| *Vita sancti Malachiae* |
| Bruno of Cologne (1030–1101) <br> *Bruno Carthusianorum* | *Expositio in epistolas Pauli* |
|━━| *Expositio in Psalmos* |
| Elisabeth of Schönau (1129–1164/5) <br> *Elisabeth Schoenaugiensis* | *Elisabethae de sacro exercitu uirginum Coloniensium* |
|━━| *Epistolae* |
|━━| *Liber primus* | 
|━━| *Liber secundus* |
|━━| *Liber tertius* |
|━━| *Liber uiarum dei* |
|━━| *Liber reuelationum* |
| Ekbert of Schönau († 1184) <br> *Ekbertus Schoenaugiensis* | *De obitu* |
|━━| *Epistolae Ekberti* |
|━━| *Laudationes* |
|━━| *Magnificat anima mea* |
|━━| *Missus est angelus* |
|━━| *Orationes* |
|━━| *Sermones XIII contra catharos* |
|━━| *Soliloquium seu meditationes* |
|━━| *Ymnus de S. Gregorio* |
| Gerhoh of Reichersberg (1093–1169) <br> *Gerhohus Reicherspergensis* | *Commentarius aureus in Psalmos et cantica ferialia* |
|━━| *De aedificio dei* |
|━━| *Epistolae Gerhohi* |
| Gilbert of Poitiers (1075–1154) <br> *Gislebertus Porretanus* | *Expositio in Boethii librum De bonorum hedbomade* |
|━━| *Liber de sex principiis* |
| Guibert of Gembloux (1124/5–1214) <br> *Guibertus Gemblacensis* | *De combustione monasterii Gemblacensis* |
|━━| *Epistolarium* |
| Guibert de Nogent (c. 1053–1125) <br> *Guibertus de Nouigento* | *Carmen Erumnarum et dolorum* |
|━━| *Contra iudaizantem et Iudaeos* |
|━━| *De bucella Iudae data* |
|━━| *De sanctis et eorum pigneribus* |
|━━| *De uita sua siue Monodiae* |
|━━| *Historia quae inscribitur Dei gesta per Francos* |
|━━| *In Zachariam* |
|━━| *Quo ordine sermo fieri debeat* |
| Heloise of Argenteuil († 1164) <br> *Heloisa Argentoliensis* | *Epistolae ad Abaelardum (II, IV and VI)* |
| Texts related to Heloise and Abelard | Tegernsee collection |
|━━| *Epistolae duorum amantium* <br> both <*V*>*ir* and <*M*>*ulier*|
| Hildebert of Lauardin (c. 1055–1133) <br> *Hildebertus Lauardinensis* | *Epistolae de Paschali papa* |
|━━| *Vita Mariae Aegyptiacae* |
| Hildegard of Bingen (1098–1179) <br> *Hildegardis Bingensis* | *Cause et cure* |
|━━| *De excellentia sancti Martini* |
|━━| *De regula Benedicti* |
|━━| *Epistolae* |
|━━| *Epistulae recensionem retractatam* |
|━━| *Explanatio euangeliorum* |
|━━| *Explanatio symboli Athanasii* |
|━━| *Liber diuinorum operum uisiones* |
|━━| *Liber uite meritorum uisiones* |
|━━| *Orationes mediationes uisiones et alia (Epistularium classis VIII) uisiones* |
|━━| *Ordo uirtutum drama* |
|━━| *Sciuias uisiones* |
|━━| *Symphonia liturgica* |
|━━| *Triginta octo quaestionum solutiones* |
|━━| *Visio ad guibertum missa* |
|━━| *Vita sancti Disibodi* |
|━━| *Vita sancti Ruperti* |
|Texts related to Hildegard| *Vita Hildegardis* |
|━━| *Varii Epistolae ad Hildegardem* |
| Honorius of Regensburg (c. 1080–c. 1154) <br> *Honorius Augustodunensis* | *De apostatis* |
|━━| *De offendiculo*|
|━━| *Elucidarium*|
|━━| *Expositio in Cantica canticorum*|
|━━| *Gemma animae*|
|━━| *Imago mundi*|
|━━| *Imago mundi: Continuatio Weingartensis*|
|━━| *Imago mundi (cum continuationibus VII) (Excerpta) Speculum ecclesiae*|
|━━| *Summa gloria* |
|━━| *Summa totius* |
| Hugh of Saint-Victor (c. 1096–1141) <br> *Hugo de Sancto Victore* | *De Sacramentis Christianae fidei* |
|━━| *De septem donis spiritus sancti* |
|━━| *De substantia dilectionis* |
|━━| *De tribus rerum subsistentiis* |
|━━| *De uanitate rerum mundanarum* |
|━━| *De uerbo dei* |
|━━| *Dialogus de creatione mundi* |
|━━| *In hierarchicam coelestem* |
|━━| *Sententiae de diuinitate* |
| Iuo of Chartres (1040–1115) <br> *Iuo Carnotensis* | *Decretum* |
|━━| *Epistolae ad litem inuestiturarum spectantes* |
|━━| *Panormia* |
| John of Salisbury (c. 1125–1180) <br> *Ioannes Saresberiensis* | *Epistolae* |
|━━| *Metalogicon* |
|━━| *Polycraticus* |
|━━| *Vita Sancti Anselmi Cantuariensis* |
| Nicholas of Montiéramey <br> *Nicolaus Claraeuallensis* | *Epistolae* |
|━━| *Sermones* |
| Odo of Deuil (1110–1162) <br> *Odo de Deogilo* | *De profectione Ludouici VII in Orientem* |
| Peter Abelard (1079–1142) <br> *Petrus Abaelardus* | *Apologia contra Bernardum* |
|━━| *Commentaria in epistulam Pauli ad Romanos* |
|━━| *Epistolae I, III, V, VII, VIII* |
|━━| *Epistolae IX–XIV* |
|━━| *Expositio in Hexameron* |
|━━| *Glossae super Hermeneias* |
|━━| *Scito te ipsum* |
|━━| *Sermones* |
|━━| *Sic et non* |
|━━| *Theologia Christiana* |
|━━| *Theologia Scholarium* |
|━━| *Theologia 'Summi boni'* |
| Peter Damian (c. 1007–c.1073) <br> *Petrus Damianus* | *De diuina omnipotentia* |
|━━| *Liber Gomorrhianus* |
|━━| *Vita sancti Romualdi* |
| Peter Lombard (c. 1096–1160) <br> *Petrus Lombardus* | *Collectanea in epistolas Pauli* |
|━━| *Commentaria in Psalmos* |
|━━| *Sententiae* |
| Peter of Celle (c. 1115–1183) <br> *Petrus Cellensis* | *De conscientia* |
|━━| *Epistolae* |
|━━| *Liber de panibus* |
|━━| *Mystica et moralis expositio Mosaici tabernaculi Sermones* |
| Peter the Venerable (1092–1156) <br> *Petrus Venerabilis* | *Aduersus Iudaeorum inueteratam duritiem* |
|━━| *Aduersus sectam Saracenorum* |
|━━| *Carmina* |
|━━| *Contra Petrobrusianos haereticos* |
|━━| *De miraculis libri duo* |
|━━| *Epistulae* |
| Rupert of Deutz (c. 1075–1130) <br> *Rupertus Tuitiensis* | *Anulus siue dialogus inter Christianum et Iudaeum* |
|━━| *Carmina de calamitatibus ecclesiae Leodiensis Commentaria in Canticum Canticorum* |
|━━| *Commentaria in duodecim prophetas minores* |
|━━| *Commentaria in euangelium sancti Iohannis Commentarium in Apocalypsim Iohannis apostoli* |
|━━| *De gloria et honore* |
|━━| *De glorificatione Trinitatis et processione Spiritus sancti De incendio Tuitiensi* |
|━━| *De meditatione mortis* |
|━━| *De sancta trinitate et operibus eius* |
|━━| *De uictoria uerbi Dei* |
|━━| *[?] Epistula ad F.* |
|━━| *Hymnus primus de sancto spiritu* |
|━━| *Hymnus secundus siue Oratio ad sanctum spiritum Liber de diuinis offciis* |
|━━| *Liber de laesione uirginitatis* |
|━━| *[?] Officium de festo Sancti Augustini* |
|━━| *Passio Eliphii Tullensis* |
|━━| *Sermo de Pantaleone* |
|━━| *[?] Vita Heriberti Coloniensis* |
| Suger of Saint-Denis (1080/1–1151) <br> *Sugerius Sancti Dionysii* | *De consecratione* |
|━━| *De rege Ludouico* |
|━━| *Epistolae* |
|━━| *Fragmentum uitae Ludouici Iunioris* |
|━━| *Gesta Sugerii (De admin.)* |
|━━| *Vita Ludouici grossi* |
| Texts related to Saint-Denis | *Cartulaire blanc* | 
|━━| D Kar 286 |
| Theoderic of Echternach <br> *Theodericus Epternacensis* | *Chronicon Epternacense* |
|━━| *Libellus de libertate Epternacensi propugnata* |
| Walter of Châtillon (c. 1135–c. 1179) <br> *Gualterus de Castellione* | *Alexandreis* |
|━━| *[?] De SS. Trinitate tractatus* |
|━━| *Tractatus contra Iudaeos* |
| William of Conches (c. 1090–1154) <br> *Guillelmus de Conchis* | *Dragmaticon Philosophiae* |
|━━| *Glosae super Boetium* |
| William of Saint-Denis <br> *Guillelmus Sancti Dionysii* | *Ad quosdam ex suis comonachis* |
|━━| *De morte Sugerii* |
|━━| *Dyalogum apologiticum* |
|━━| *Vita Sugerii* |
| William of Saint-Thièrry (c. 1080–1148) <br> *Guillelmus de Sancto Theodorico* | *Aenigma fidei* |
|━━| *Breuis commentatio (in Cantici Canticorum priora dua capita)* |
|━━| *De contemplando Deo* |
|━━| *Epistola de erroribus Guillelmi de Conchis* |
|━━| *De natura corporis et animae* |
|━━| *De natura et dignitate amoris* |
|━━| *De sacramento altaris* |
|━━| *Disputatio aduersus Abaelardum* |
|━━| *[?] Disputatio altera aduersus Abaelardum* |
|━━| *Expositio altera super Cantica canticorum* |
|━━| *Expositio in Epistolam ad Romanos* |
|━━| *Orationes meditatiuae* |
|━━| *Speculum fidei* |

# Acknowledgements

I would like to express my gratitude to my supervisors, prof. dr. Jeroen Deploige, prof. dr. Wim Verbaal and prof. dr. Mike Kestemont.
Furthermore, I want to thank the [Ghent University Special Research Fund (BOF)](https://www.ugent.be/en/research/funding/bof), the [Research Foundation Flanders (FWO)](https://www.fwo.be/en/), the [Henri Pirenne Institute for Medieval Studies at Ghent University](https://www.ugent.be/pirenne/en), the [CLiPS Computational Linguistics Group at the University of Antwerp](https://www.clips.uantwerpen.be/computational-linguistics?page=4), and the *Centre Traditio Litterarum Occidentalium* division for computer-assisted research into Latin language and literature housed in the [*Corpus Christianorum* Library and Knowledge Centre](https://www.corpuschristianorum.org/) of Brepols Publishers in Turnhout (Belgium).

# Further Reference

* Kestemont, Mike, Sara Moens, and Jeroen Deploige. ‘Collaborative Authorship in the Twelfth Century: A Stylometric Study of Hildegard of Bingen and Guibert of Gembloux'. *Digital Scholarship in the Humanities* 30, no. 2 (2013): 199–224.
* De Gussem, Jeroen. 'Bernard of Clairvaux and Nicholas of Montiéramey: Tracing the Secretarial Trail with Computational Stylistics'. *Speculum* 92, no. S1 (2017): S190–S225.
* Kestemont, Mike, and Jeroen De Gussem. 'Integrated Sequence Tagging for Medieval Latin Using Deep Representation Learning'. Edited by Marco Büchler and Laurence Mellerin. *Journal of Data Mining and Digital Humanities*, Special Issue on Computer-Aided Processing of Intertextuality in Ancient Languages, 2017, 1–17. [https://arxiv.org/abs/1603.01597](https://arxiv.org/abs/1603.01597).
* De Gussem, Jeroen, and Dinah Wouters. ‘Language and Thought in Hildegard of Bingen's Visionary Trilogy: Close and Distant Readings of a Thinker's Development'. *Parergon*, 2019, 31–60.