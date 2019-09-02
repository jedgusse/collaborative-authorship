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
| Alan of Lille (c. 1128–c. 1203) *Alanus de Insulis* | *Anticlaudianus* |
|—|*Contra haereticos*|
|—|*De arte praedicatoria*|
|—|*De planctu naturae*|
|—|*Elucidatio in Cantica canticorum*|
|—|*Sermones*|
|—|*Summa "Quoniam homines"*|
| Anselm of Canterbury (1033–1109) *Anselmus Cantuariensis* | *Cur deus homo*|
|—|*Monologion*|
|—|*Proslogion*|


# Code

# Acknowledgements

I would like to express my gratitude to my supervisors, prof. dr. Jeroen Deploige, prof. dr. Wim Verbaal and prof. dr. Mike Kestemont.
Furthermore, I want to thank the [Ghent University Special Research Fund (BOF)](https://www.ugent.be/en/research/funding/bof), the [Research Foundation Flanders (FWO)](https://www.fwo.be/en/), the [Henri Pirenne Institute for Medieval Studies at Ghent University](https://www.ugent.be/pirenne/en), the [CLiPS Computational Linguistics Group at the University of Antwerp](https://www.clips.uantwerpen.be/computational-linguistics?page=4), and the *Centre Traditio Litterarum Occidentalium* division for computer-assisted research into Latin language and literature housed in the [*Corpus Christianorum Library and Knowledge Centre*](https://www.corpuschristianorum.org/) of Brepols Publishers in Turnhout (Belgium).