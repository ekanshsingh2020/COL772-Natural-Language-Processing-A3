from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import csv
import torch
import json
import tqdm
import sys,os
import re
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
os.environ['CURL_CA_BUNDLE'] = ''
def prompt_generate(article):
    
    prompt = ""
    prompt += f"Write the highlights of this article for a layman: \n {article}\n\n"
    return prompt

model_dir=sys.argv[2]
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base",unk_token="<unk>",bos_token="<s>",eos_token="</s>",pad_token = "<s>",padding_side = 'left')
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, torch_dtype=torch.float16,).to('cuda')

result_dir=sys.argv[3]
# open a .txt file in write mode if not exists it will create a new one
eLife_outfile = open(os.path.join(result_dir,'elife.txt'), 'w')
# csvwriter = csv.writer(eLife_outfile)
# csvwriter.writerow(['summary'])

data_dir=sys.argv[1]
eLife_train_data = [json.loads(line) for line in open(os.path.join(data_dir,'eLife_train.jsonl'))]
eLife_test_data = [json.loads(line) for line in open(os.path.join(data_dir,'eLife_test.jsonl'))]


example_prompt = prompt_generate("Kidney function depends on the nephron , which comprises a blood filter , a tubule that is subdivided into functionally distinct segments , and a collecting duct . How these regions arise during development is poorly understood . The zebrafish pronephros consists of two linear nephrons that develop from the intermediate mesoderm along the length of the trunk . Here we show that , contrary to current dogma , these nephrons possess multiple proximal and distal tubule domains that resemble the organization of the mammalian nephron . We examined whether pronephric segmentation is mediated by retinoic acid ( RA ) and the caudal ( cdx ) transcription factors , which are known regulators of segmental identity during development . Inhibition of RA signaling resulted in a loss of the proximal segments and an expansion of the distal segments , while exogenous RA treatment induced proximal segment fates at the expense of distal fates . Loss of cdx function caused abrogation of distal segments , a posterior shift in the position of the pronephros , and alterations in the expression boundaries of raldh2 and cyp26a1 , which encode enzymes that synthesize and degrade RA , respectively . These results suggest that the cdx genes act to localize the activity of RA along the axis , thereby determining where the pronephros forms . Consistent with this , the pronephric-positioning defect and the loss of distal tubule fate were rescued in embryos doubly-deficient for cdx and RA . These findings reveal a novel link between the RA and cdx pathways and provide a model for how pronephric nephrons are segmented and positioned along the embryonic axis .  The kidney eliminates metabolic waste in the body using highly specialized structures called nephrons . Individual nephrons are composed of a blood filter ( renal corpuscle ) , a tubule that recovers or secretes solutes , and a collecting duct [1] . The renal corpuscle contains epithelial cells called podocytes that form the slit-diaphragm filtration barrier and allow collection of substances from the blood [2] . In a number of vertebrate species , including some mammals , the renal corpuscle is connected to the tubule by a short stretch of ciliated epithelium called the neck segment that guides filtrate entry into the tubule [3–5] . The mammalian nephron tubule is subdivided into a series of proximal and distal segments connected to a collecting duct [1 , 6] . The polarized epithelial cells in the tubule segments have a unique ultrastructure and express a select cohort of solute transporters [1] . Thus , each segment is functionally distinct and performs the transport of particular solutes that are required for proper renal function . In higher vertebrates , three kidneys of increasing complexity arise sequentially from the intermediate mesoderm ( IM ) : the pronephros , the mesonephros , and the metanephros [7] . The pronephros and mesonephros degenerate in succession , with the metanephros serving as the adult kidney . Lower vertebrates , such as fish and amphibians , develop a pronephros during embryonic stages , and then form a mesonephros that will be used throughout their adult life [8–10] . Each of these kidneys contains the nephron as its basic functional unit [8] . To date , much of our knowledge of kidney development has come from gene-targeting studies in the mouse [7 , 11 , 12] . These experiments have identified a number of genes that play essential roles in the early stages of metanephros development , but there is a limited understanding of the molecular pathways governing the later stages of kidney ontogeny , when individual nephrons form and become segmented [7] . The zebrafish is an ideal genetic and developmental model system for dissecting the molecular mechanisms of nephron formation because of the anatomical simplicity of the pronephros , which contains two nephrons as opposed to the thousands of nephrons in a mammalian metanephros.")
example_prompt += f"\n\n ## Summary for a layman: In the kidney , structures known as nephrons are responsible for collecting metabolic waste . Nephrons are composed of a blood filter ( glomerulus ) followed by a series of specialized tubule regions , or segments , which recover solutes such as salts , and finally terminate with a collecting duct . The genetic mechanisms that establish nephron segmentation in mammals have been a challenge to study because of the kidney's complex organogenesis . The zebrafish embryonic kidney ( pronephros ) contains two nephrons , previously thought to consist of a glomerulus , short tubule , and long stretch of duct . In this study , we have redefined the anatomy of the zebrafish pronephros and shown that the duct is actually subdivided into distinct tubule segments that are analogous to the proximal and distal segments found in mammalian nephrons . Next , we used the zebrafish pronephros to investigate how nephron segmentation occurs .\n\n"
# print(example_prompt)
total_length=0
for ind,sample in enumerate(tqdm.tqdm(eLife_test_data)):
    total_length+=1

cnt=0
for ind,sample in enumerate(tqdm.tqdm(eLife_test_data)):
    
    prompt = example_prompt + prompt_generate('\n'.join(sample['article'].split('\n')[0:2]))
    prompt += "\n\n ## Summary for a layman: "
    prompts = [prompt]
    input_ids = tokenizer(prompt, return_tensors = "pt")
    generated_ids = model.generate(input_ids = input_ids['input_ids'].to('cuda'),repetition_penalty = 4.5 ,min_length = 300, max_length = 1024)
    responses = tokenizer.batch_decode(generated_ids, skip_special_tokens = True)
    predicted_answer = responses[0].strip().replace("\n"," ")
    predicted_answer=re.sub(r"http\S+|www\S+|https\S+", "", predicted_answer)
    # csvwriter.writerow([predicted_answer])
    eLife_outfile.write(predicted_answer)
    cnt+=1
    if cnt<total_length:
        eLife_outfile.write('\n')
    
# close the file
print('eLife Closing the file...')
eLife_outfile.close()
print("eLife File Saved")

# open a .txt file in write mode if not exists it will create a new one
PLOS_outfile = open(os.path.join(result_dir,'plos.txt'), 'w')
# csvwriter = csv.writer(PLOS_outfile)
# csvwriter.writerow(['summary'])

data_dir=sys.argv[1]
PLOS_train_data = [json.loads(line) for line in open(f'{data_dir}/PLOS_train.jsonl')]
PLOS_test_data = [json.loads(line) for line in open(f'{data_dir}/PLOS_test.jsonl')]


example_prompt = prompt_generate('In temperate climates , winter deaths exceed summer ones . However , there is limited information on the timing and the relative magnitudes of maximum and minimum mortality , by local climate , age group , sex and medical cause of death . We used geo-coded mortality data and wavelets to analyse the seasonality of mortality by age group and sex from 1980 to 2016 in the USA and its subnational climatic regions . Death rates in men and women ≥ 45 years peaked in December to February and were lowest in June to August , driven by cardiorespiratory diseases and injuries . In these ages , percent difference in death rates between peak and minimum months did not vary across climate regions , nor changed from 1980 to 2016 . Under five years , seasonality of all-cause mortality largely disappeared after the 1990s . In adolescents and young adults , especially in males , death rates peaked in June/July and were lowest in December/January , driven by injury deaths .  It is well-established that death rates vary throughout the year , and in temperate climates there tend to be more deaths in winter than in summer ( Campbell , 2017; Fowler et al . , 2015; Healy , 2003; McKee , 1989 ) . It has therefore been hypothesized that a warmer world may lower winter mortality in temperate climates ( Langford and Bentham , 1995; Martens , 1998 ) . In a large country like the USA , which possesses distinct climate regions , the seasonality of mortality may vary geographically , due to geographical variations in mortality , localized weather patterns , and regional differences in adaptation measures such as heating , air conditioning and healthcare ( Davis et al . , 2004; Braga et al . , 2001; Kalkstein , 2013; Medina-Ramón and Schwartz , 2007 ) . The presence and extent of seasonal variation in mortality may also itself change over time ( Bobb et al . , 2014; Carson et al . , 2006; Seretakis et al . , 1997; Sheridan et al . , 2009 ) . A thorough understanding of the long-term dynamics of seasonality of mortality , and its geographical and demographic patterns , is needed to identify at-risk groups , plan responses at the present time as well as under changing climate conditions . Although mortality seasonality is well-established , there is limited information on how seasonality , including the timing of minimum and maximum mortality , varies by local climate and how these features have changed over time , especially in relation to age group , sex and medical cause of death ( Rau , 2004; Rau et al . , 2018 ) . In this paper , we comprehensively characterize the spatial and temporal patterns of all-cause and cause-specific mortality seasonality in the USA by sex and age group , through the application of wavelet analytical techniques , to over three decades of national mortality data . Wavelets have been used to study the dynamics of weather phenomena ( Moy et al . , 2002 ) and infectious diseases ( Grenfell et al . , 2001 ) . We also used centre of gravity analysis and circular statistics methods to understand the timing of maximum and minimum mortality . In addition , we identify how the percentage difference between death rates in maximum and minimum mortality months has changed over time .')
example_prompt += f"\n\n ## Summary for a layman: In the USA , more deaths happen in the winter than the summer . But when deaths occur varies greatly by sex , age , cause of death , and possibly region . Seasonal differences in death rates can change over time due to changes in factors that cause disease or affect treatment . Analyzing the seasonality of deaths can help scientists determine whether interventions to minimize deaths during a certain time of year are needed , or whether existing ones are effective . Scrutinizing seasonal patterns in death over time can also help scientists determine whether large-scale weather or climate changes are affecting the seasonality of death . Now , Parks et al . show that there are age and sex differences in which times of year most deaths occur . Parks et al . analyzed data on US deaths between 1980 and 2016 . While overall deaths in a year were highest in winter and lowest in summer , a greater number of young men died during summer – mainly due to injuries – than during winter . Seasonal differences in deaths among young children have largely disappeared and seasonal differences in the deaths of older children and young adults have become smaller . Deaths among women and men aged 45 or older peaked between December and February – largely caused by respiratory and heart diseases , or injuries . Deaths in this older age group were lowest during the summer months . Death patterns in older people changed little over time . No regional differences were found in seasonal death patterns , despite large climate variation across the USA .\n\n"
# print(example_prompt)

total_length=0
for ind,sample in enumerate(tqdm.tqdm(PLOS_test_data)):
    total_length+=1

cnt=0
for ind,sample in enumerate(tqdm.tqdm(PLOS_test_data)):
    
    prompt = example_prompt + prompt_generate('\n'.join(sample['article'].split('\n')[0:2]))
    prompt += "\n\n ## Summary for a layman: "
    prompts = [prompt]
    input_ids = tokenizer(prompt, return_tensors = "pt")
    generated_ids = model.generate(input_ids = input_ids['input_ids'].to('cuda'),repetition_penalty = 4.5 ,min_length = 300, max_length = 1024)
    responses = tokenizer.batch_decode(generated_ids, skip_special_tokens = True)
    predicted_answer = responses[0].strip().replace("\n"," ")
    predicted_answer=re.sub(r"http\S+|www\S+|https\S+", "", predicted_answer)
    # csvwriter.writerow([predicted_answer])
    PLOS_outfile.write(predicted_answer)
    cnt+=1
    if cnt<total_length:
        PLOS_outfile.write('\n')
    
# close the file
print('PLOS Closing the file...')
PLOS_outfile.close()
print("PLOSFile Saved")