## !pip install -q git+https://github.com/huggingface/transformers.git
## !pip install ohmeow-blurr -q
## !pip install bert-score -q
## !pip install pandas

import pandas as pd
from fastai.text.all import *
from transformers import *
from blurr.data.all import *
from blurr.modeling.all import *

#Get data
df = pd.read_csv('articles.csv', error_bad_lines=False, sep=',')
df = df.dropna().reset_index()

#Select part of data we want to keep
df = df[(df['language']=='english') & (df['type']=='bs')].reset_index()
df = df[['title','text']]

#Clean text
df['text'] = df['text'].apply(lambda x: x.replace('\n',''))

#Select only part of it (makes testing faster)
articles = df.head(100)
articles.head()

#Import the pretrained model
pretrained_model_name = "facebook/bart-large-cnn"
hf_arch, hf_config, hf_tokenizer, hf_model = BLURR.get_hf_objects(pretrained_model_name, 
                                                                  model_cls=BartForConditionalGeneration)

#Create mini-batch and define parameters
hf_batch_tfm = HF_Seq2SeqBeforeBatchTransform(hf_arch, hf_config, hf_tokenizer, hf_model, 
    task='summarization',
    text_gen_kwargs=
 {'max_length': 248,'min_length': 56,'do_sample': False, 'early_stopping': True, 'num_beams': 4, 'temperature': 1.0, 
  'top_k': 50, 'top_p': 1.0, 'repetition_penalty': 1.0, 'bad_words_ids': None, 'bos_token_id': 0, 'pad_token_id': 1,
 'eos_token_id': 2, 'length_penalty': 2.0, 'no_repeat_ngram_size': 3, 'encoder_no_repeat_ngram_size': 0,
 'num_return_sequences': 1, 'decoder_start_token_id': 2, 'use_cache': True, 'num_beam_groups': 1,
 'diversity_penalty': 0.0, 'output_attentions': False, 'output_hidden_states': False, 'output_scores': False,
 'return_dict_in_generate': False, 'forced_bos_token_id': 0, 'forced_eos_token_id': 2, 'remove_invalid_values': False})


#Prepare data for training
blocks = (HF_Seq2SeqBlock(before_batch_tfm=hf_batch_tfm), noop)
dblock = DataBlock(blocks=blocks, get_x=ColReader('text'), get_y=ColReader('title'), splitter=RandomSplitter())
dls = dblock.dataloaders(articles, batch_size = 2)

#Define performance metrics
seq2seq_metrics = {
        'rouge': {
            'compute_kwargs': { 'rouge_types': ["rouge1", "rouge2", "rougeL"], 'use_stemmer': True },
            'returns': ["rouge1", "rouge2", "rougeL"]
        },
        'bertscore': {
            'compute_kwargs': { 'lang': 'fr' },
            'returns': ["precision", "recall", "f1"]}}

#Model
model = HF_BaseModelWrapper(hf_model)
learn_cbs = [HF_BaseModelCallback]
fit_cbs = [HF_Seq2SeqMetricsCallback(custom_metrics=seq2seq_metrics)]

#Specify training
learn = Learner(dls, model,
                opt_func=ranger,loss_func=CrossEntropyLossFlat(),
                cbs=learn_cbs,splitter=partial(seq2seq_splitter, arch=hf_arch)).to_fp16()

#Create optimizer with default hyper-parameters
learn.create_opt() 
learn.freeze()

#Training
learn.fit_one_cycle(10, lr_max=3e-5, cbs=fit_cbs)

#Texts for BART
fiji_text = "The researchers examined three types of coral in reefs off the coast of Fiji ... The researchers found when fish were plentiful,\
they would eat algae and seaweed off the corals, which appeared\
to leave them more resistant to the bacterium Vibrio coralliilyticus,\
a bacterium associated with bleaching. The researchers suggested the algae, like warming temperatures, might render the\
corals’ chemical defenses less effective, and the fish were protecting the coral by removing the algae."

diplomat_text = "in a traffic collision ... Prime Minister Johnson was questioned\
about the case while speaking to the press at a hospital in Watford.\
He said, “I hope that Anne Sacoolas will come back ...\
if we can’t resolve it then of course I will be raising it myself\
personally with the White House.”"

covid_text = "The ongoing outbreak of the novel coronavirus disease 2019 (COVID-19) originating from Wuhan, China, \
             draws worldwide concerns due to its long incubation period and strong infectivity. Although RT-PCR-based molecular diagnosis \
             techniques are being widely applied for clinical diagnosis currently, timely and accurate diagnosis are still limited due to \
             labour intensive and time-consuming operations of these techniques. To address the issue, herein we report the synthesis of \
             poly (amino ester) with carboxyl groups (PC)-coated magnetic nanoparticles (pcMNPs), and the development of pcMNPs-based viral\
             RNA extraction method for the sensitive detection of COVID-19 causing virus, the SARS-CoV-2. This method combines the lysis \
             and binding steps into one step, and the pcMNPs-RNA complexes can be directly introduced into subsequent RT-PCR reactions. \
             The simplified process can purify viral RNA from multiple samples within 20 min using a simple manual method or an automated \
             high-throughput approach. By identifying two different regions (ORFlab and N gene) of viral RNA, a 10-copy sensitivity \
             and a strong linear correlation between 10 and 105 copies of SARS-CoV-2 pseudovirus particles are achieved. \
             Benefitting from the simplicity and excellent performances, this new extraction method can dramatically reduce the turn-around time \
             and operational requirements in current molecular diagnosis of COVID-19, in particular for the early clinical diagnosis."


another_covid_text = "Migration flows from Latin American, African and Asian countries to Europe have shown that a high \
percentage of arriving individuals may be chronically infected with Strongyloides stercoralis, which may have a public \
health impact in non-endemic countries that are hosting these populations [1,2]. The infection is ubiquitous in tropical \
and subtropical areas, although it may also occur in temperate countries with appropriate conditions such as certain areas \
of Spain or Italy [3,4,5]. Worldwide estimates based on standard fecal techniques have suggested that between 30 and 100 \
million people are infected worldwide. These figures may be underestimates due to the low sensitivity of traditional diagnostic methods [1,6].\
Unlike other parasitic infections, this helminth has some characteristics that are of particular importance for migrant populations [7]. \
Firstly, the infection can persist for the whole lifetime due to the possibility of causing autoinfection in the human host [8]. \
Through this phenomenon, the filariform larvae penetrate intestinal mucosa in the large intestine or perianal skin and migrate \
to complete another lifecycle. Therefore, people coming from endemic areas may be at risk for their whole life, irrespective \
of the moment they arrive in a non-endemic area, as long as they are not treated. Secondly, Strongyloides stercoralis infection \
is generally asymptomatic or causes unspecific symptoms, and thus goes unnoticed by health professionals who are not looking for it [9]. \
Thirdly, although the infection is rarely transmitted from person to person [10], it can be transmitted through organ transplantation, \
and autochthonous cases have been reported in non-endemic areas [11]. Therefore, screening should be considered for potential donors at risk \
of the infection [12,13,14]. Finally, in the case of immunosuppression, particularly those displaying a concomitant use of steroids, \
transplant recipients, or patients with malignancies and Human T-Cell Lymphotropic virus-1 co-infections, the parasite may enter into\
a high replicating cycle (called hyperinfection) or disseminate to vital organs (disseminated strongyloidiasis), causing a severe disease \
with a high mortality [15]. The diagnosis of strongyloidiasis in non-endemic areas is currently based on a serological test,\
which has a considerably higher sensitivity compared with standard fecal techniques [8]. \
    Despite having cross-reactions with other helminthic infections, this is less likely to occur in migrant populations since\
    the possibility of co-infections is lower [16] and it is thus now the current recommended screening technique for these populations [17]. \
    The sensitivity of the serological tests in immunosuppressed individuals seems to be lower [18], but only limited data are available and \
    further prospective studies should better evaluate the accuracy of serological tests in immunosuppressed patients.\
    Even with the limitations of current diagnostic methods, the screening of high-risk groups and treatment of infected individuals are of\
    key importance. In this regard, the screening of strongyloidiasis in newly arrived migrants has been recommended by the European \
    Centre for Disease Prevention and Control [19,20], particularly in immunosuppressed individuals, given the potential individual \
    morbidity and mortality [16]. Evidence of the seroprevalence of strongyloidiasis in migrant populations is scarce, particularly if \
    it is assessed in Asian countries. Available data suggest that it is known to vary substantially, depending on the country of origin, \
    being particularly high in people coming from countries such as Cambodia (36%) or Latin American countries (26%) [21]. \
    Hospital-based prevalence studies conducted in specialized units have suggested that the prevalence of S.stercoralis is between 4.5% \
    and 11% in migrant populations [22,23,24]. Available data suggest a higher prevalence of infection in immunosuppressed individuals at risk [25,26,27].\
Our study aims to evaluate the prevalence of S. stercoralis at the hospital level in migrant populations or long-term travelers \
being attended to in out-patient and in-patient units as part of the systematic screening implemented in six Spanish hospitals."

#Inference (text generation)
outputs = learn.blurr_generate(fiji_text, early_stopping=False, num_return_sequences=1)

for idx, o in enumerate(outputs):
    print(f'=== Prediction {idx+1} ===\n{o}\n')

print('\n' + '*'*50 + '\n')

outputs = learn.blurr_generate(diplomat_text, early_stopping=False, num_return_sequences=1)

for idx, o in enumerate(outputs):
    print(f'=== Prediction {idx+1} ===\n{o}\n')

print('\n' + '*'*50 + '\n')

outputs = learn.blurr_generate(covid_text, early_stopping=False, num_return_sequences=1)

for idx, o in enumerate(outputs):
    print(f'=== Prediction {idx+1} ===\n{o}\n')

print('\n' + '*'*50 + '\n')

outputs = learn.blurr_generate(another_covid_text, early_stopping=False, num_return_sequences=1)

for idx, o in enumerate(outputs):
    print(f'=== Prediction {idx+1} ===\n{o}\n')

print('\n' + '*'*50 + '\n')
