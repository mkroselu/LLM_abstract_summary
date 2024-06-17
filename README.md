**Objectives**: This project aims to summarize research articles of remote sensing, which is trained by abstracts and titles of papers in the field of remote sensing. The training data was downloaded for recent years from 2020 to 2024 and the transformers were used for model training and text generation. However, this is the first attempt of utilizing LLM to generate contents for remote sensing papers. Further model fine-tuning is required to increase the accuracy of the outputs and the practicality of the project.  

# Step-by-Step Guide to Curate Data and Fine-Tune a Model 

## Step 1: Collect and Curate Data 
You can manually collect data from various sources such as scientific journals, research papers, and news articles, and organize it into a CSV file. 


```python
import pandas as pd

# Example data
data = {
    'title': [
        'AI in Remote Sensing: Current Trends',
        'Multi-Sensor Data Integration for Enhanced Geospatial Analysis',
        'Satellite Imagery for Environmental Monitoring',
        'Advances in Hyperspectral Imaging for Agriculture'
    ],
    'abstract': [
        'This paper explores the use of artificial intelligence in remote sensing and its current trends.',
        'The integration of data from multiple sensors can provide better insights for geospatial analysis.',
        'Satellite imagery is increasingly used for environmental monitoring and management.',
        'Hyperspectral imaging advancements are improving agricultural monitoring and yield prediction.'
    ],
    'keywords': [
        'AI, remote sensing, trends',
        'multi-sensor, data integration, geospatial analysis',
        'satellite imagery, environmental monitoring',
        'hyperspectral imaging, agriculture, monitoring'
    ]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('remote_sensing_dataset.csv', index=False)

```

## Step 2: Fine-Tune the Model with HuggingFace
Use the HuggingFace transformers library to fine-tune a language model on the curated dataset. 


```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, AutoTokenizer, AutoConfig 
import torch
from datasets import load_dataset 

dataset = load_dataset("csv", data_files={"train": "full_abstract_train.csv", "validation": "full_abstract_val.csv", "test": "full_abstract_test.csv"})
dataset
```




    DatasetDict({
        train: Dataset({
            features: ['title', 'abstract'],
            num_rows: 251
        })
        validation: Dataset({
            features: ['title', 'abstract'],
            num_rows: 31
        })
        test: Dataset({
            features: ['title', 'abstract'],
            num_rows: 32
        })
    })




```python
# Load model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Add padding token if not present
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Resize model embeddings to accommodate the new padding token
model.resize_token_embeddings(len(tokenizer))
```




    Embedding(50258, 768)




```python
# Tokenize function
def tokenize_function(examples):
    return tokenizer(examples["abstract"], truncation=True, padding="max_length", max_length=512)

# Tokenize dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)
```


    Map:   0%|          | 0/251 [00:00<?, ? examples/s]



    Map:   0%|          | 0/31 [00:00<?, ? examples/s]



    Map:   0%|          | 0/32 [00:00<?, ? examples/s]



```python
dataset['train']['title'][0]
```




    'A review of remote sensing for environmental monitoring in China'




```python
example_ids = tokenized_datasets['train'][0]['input_ids']
print(f"Example token IDs: {example_ids}") 
```

    Example token IDs: [464, 3288, 2858, 318, 6393, 329, 1692, 9441, 290, 2478, 1201, 340, 3769, 1660, 4133, 11, 1956, 4133, 11, 10685, 4133, 290, 4258, 4133, 3503, 13, 1081, 257, 5922, 1499, 11, 2807, 468, 13923, 257, 2383, 1487, 287, 262, 3288, 2858, 287, 2274, 4647, 26, 290, 4361, 11, 9904, 290, 45116, 262, 3722, 286, 262, 2858, 318, 286, 1049, 12085, 13, 14444, 284, 262, 9695, 286, 1588, 12, 9888, 290, 8925, 13432, 11, 6569, 34244, 3037, 468, 587, 281, 35669, 3164, 329, 6142, 9904, 13, 770, 3348, 8088, 262, 11210, 4133, 11, 6712, 290, 4788, 329, 6142, 9904, 287, 2807, 11, 290, 262, 14901, 287, 2267, 290, 3586, 286, 6569, 34244, 422, 1936, 7612, 25, 25047, 6376, 45069, 11, 6142, 9904, 287, 6861, 3006, 11, 10016, 3006, 11, 7876, 3006, 290, 9691, 3006, 13, 383, 6569, 34244, 4981, 290, 5050, 329, 2972, 3858, 286, 6142, 9904, 11, 290, 262, 2176, 5479, 287, 2807, 389, 8569, 2280, 31880, 13, 770, 3348, 635, 2173, 503, 1688, 6459, 4683, 379, 262, 1459, 3800, 25, 11210, 12694, 2761, 11, 11521, 779, 6459, 286, 40522, 11, 13479, 287, 262, 45069, 1429, 286, 25047, 9633, 11, 20796, 1245, 2761, 11, 257, 1877, 4922, 286, 22771, 11, 262, 4939, 2694, 286, 41164, 290, 9815, 3781, 11, 290, 257, 3092, 286, 31350, 1176, 329, 4858, 40522, 13, 9461, 11, 262, 2478, 5182, 290, 2003, 11678, 389, 1234, 2651, 284, 1277, 262, 2267, 290, 3586, 286, 6142, 9904, 290, 4800, 287, 262, 649, 6980, 13, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257]
    


```python
print(f"Model vocab size: {model.config.vocab_size}")
print(f"Tokenizer vocab size: {len(tokenizer)}")
```

    Model vocab size: 50258
    Tokenizer vocab size: 50258
    


```python
CONTEXT_LENGTH = 512 
config = AutoConfig.from_pretrained( 
    "gpt2",
    vocab_size = len(tokenizer), 
    n_ctx = CONTEXT_LENGTH, 
    bos_token_id = tokenizer.bos_token_id, 
    eos_token_id = tokenizer.eos_token_id
) 
```

    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\huggingface_hub\file_download.py:1132: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
      warnings.warn(
    


```python
model = GPT2LMHeadModel(config)
model_size = sum(t.numel() for t in model.parameters()) 
print(f"GPT-2 size:{model_size/1000**2:.1f}M parameters") 
```

    GPT-2 size:124.4M parameters
    


```python
config # what we use to create a model 
```




    GPT2Config {
      "_name_or_path": "gpt2",
      "activation_function": "gelu_new",
      "architectures": [
        "GPT2LMHeadModel"
      ],
      "attn_pdrop": 0.1,
      "bos_token_id": 50256,
      "embd_pdrop": 0.1,
      "eos_token_id": 50256,
      "initializer_range": 0.02,
      "layer_norm_epsilon": 1e-05,
      "model_type": "gpt2",
      "n_ctx": 512,
      "n_embd": 768,
      "n_head": 12,
      "n_inner": null,
      "n_layer": 12,
      "n_positions": 1024,
      "reorder_and_upcast_attn": false,
      "resid_pdrop": 0.1,
      "scale_attn_by_inverse_layer_idx": false,
      "scale_attn_weights": true,
      "summary_activation": null,
      "summary_first_dropout": 0.1,
      "summary_proj_to_labels": true,
      "summary_type": "cls_index",
      "summary_use_proj": true,
      "task_specific_params": {
        "text-generation": {
          "do_sample": true,
          "max_length": 50
        }
      },
      "transformers_version": "4.41.2",
      "use_cache": true,
      "vocab_size": 50258
    }




```python
from transformers import DataCollatorForLanguageModeling

tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm = False) 
```

## Training Model


```python
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    auto_find_batch_size = True, 
    num_train_epochs = 2,
    gradient_accumulation_steps = 8, 
    weight_decay = 0.1,
    lr_scheduler_type = "cosine", 
    learning_rate=5e-4, # 2e-5
    fp16 = True, 
    logging_steps = 10 
    # per_device_train_batch_size=16,


    ) 
```

    C:\Users\MEI-KUEI LU\anaconda3\envs\test\lib\site-packages\transformers\training_args.py:1474: FutureWarning: `evaluation_strategy` is deprecated and will be removed in version 4.46 of 🤗 Transformers. Use `eval_strategy` instead
      warnings.warn(
    


```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"], 
    data_collator = data_collator, 
    tokenizer = tokenizer 
) 
```


```python
trainer.train()
```


  <thead>
 <tr style="text-align: left;">
      <th>Epoch</th>
      <th>Training Loss</th>
      <th>Validation Loss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>No log</td>
      <td>9.320814</td>
    </tr>
    <tr>
      <td>2</td>
      <td>No log</td>
      <td>6.519324</td>
    </tr>
  </tbody>
</table><p>





    TrainOutput(global_step=8, training_loss=9.724644660949707, metrics={'train_runtime': 2458.6672, 'train_samples_per_second': 0.204, 'train_steps_per_second': 0.003, 'total_flos': 131168600064000.0, 'train_loss': 9.724644660949707, 'epoch': 2.0})




```python
# Save the fine-tuned model
model.save_pretrained("fine_tuned_model_full_abstract")
tokenizer.save_pretrained("fine_tuned_model_full_abstract") 
```




    ('fine_tuned_model_full_abstract\\tokenizer_config.json',
     'fine_tuned_model_full_abstract\\special_tokens_map.json',
     'fine_tuned_model_full_abstract\\vocab.json',
     'fine_tuned_model_full_abstract\\merges.txt',
     'fine_tuned_model_full_abstract\\added_tokens.json')



## Using Our Model In Pipeline  


```python
import torch 
from transformers import pipeline 

pipe = pipeline( 
    "text-generation", model = "fine_tuned_model_full_abstract")

```


```python
sample = dataset['test'][2]
sample 
```




    {'title': 'Assessment of land use land cover changes and future predictions using CA-ANN simulation for selangor, Malaysia',
     'abstract': 'Land use land cover (LULC) has altered dramatically because of anthropogenic activities, particularly in places where climate change and population growth are severe. The geographic information system (GIS) and remote sensing are widely used techniques for monitoring LULC changes. This study aimed to assess the LULC changes and predict future trends in Selangor, Malaysia. The satellite images from 1991–2021 were classified to develop LULC maps using support vector machine (SVM) classification in ArcGIS. The image classification was based on six different LULC classes, i.e., (i) water, (ii) developed, (iii) barren, (iv) forest, (v) agriculture, and (vi) wetlands. The resulting LULC maps illustrated the area changes from 1991 to 2021 in different classes, where developed, barren, and water lands increased by 15.54%, 1.95%, and 0.53%, respectively. However, agricultural, forest, and wetlands decreased by 3.07%, 14.01%, and 0.94%, respectively. The cellular automata-artificial neural network (CA-ANN) technique was used to predict the LULC changes from 2031–2051. The percentage of correctness for the simulation was 82.43%, and overall kappa value was 0.72. The prediction maps from 2031–2051 illustrated decreasing trends in (i) agricultural by 3.73%, (ii) forest by 1.09%, (iii) barren by 0.21%, (iv) wetlands by 0.06%, and (v) water by 0.04% and increasing trends in (vi) developed by 5.12%. The outcomes of this study provide crucial knowledge that may help in developing future sustainable planning and management, as well as assist authorities in making informed decisions to improve environmental and ecological conditions.'}




```python
prompt = f"ABSTRACT:{sample['abstract']}\n\nTITLE:"
pipe(prompt, max_new_tokens = 128) 
```




    [{'generated_text': 'ABSTRACT:Land use land cover (LULC) has altered dramatically because of anthropogenic activities, particularly in places where climate change and population growth are severe. The geographic information system (GIS) and remote sensing are widely used techniques for monitoring LULC changes. This study aimed to assess the LULC changes and predict future trends in Selangor, Malaysia. The satellite images from 1991–2021 were classified to develop LULC maps using support vector machine (SVM) classification in ArcGIS. The image classification was based on six different LULC classes, i.e., (i) water, (ii) developed, (iii) barren, (iv) forest, (v) agriculture, and (vi) wetlands. The resulting LULC maps illustrated the area changes from 1991 to 2021 in different classes, where developed, barren, and water lands increased by 15.54%, 1.95%, and 0.53%, respectively. However, agricultural, forest, and wetlands decreased by 3.07%, 14.01%, and 0.94%, respectively. The cellular automata-artificial neural network (CA-ANN) technique was used to predict the LULC changes from 2031–2051. The percentage of correctness for the simulation was 82.43%, and overall kappa value was 0.72. The prediction maps from 2031–2051 illustrated decreasing trends in (i) agricultural by 3.73%, (ii) forest by 1.09%, (iii) barren by 0.21%, (iv) wetlands by 0.06%, and (v) water by 0.04% and increasing trends in (vi) developed by 5.12%. The outcomes of this study provide crucial knowledge that may help in developing future sustainable planning and management, as well as assist authorities in making informed decisions to improve environmental and ecological conditions.\n\nTITLE: 16 terrestrial the (- neural the., the in\nGROUND the data (GROUND, maps.ation'}]




```python
prompt = f"ABSTRACT:{sample['abstract']}"
pipe(prompt, max_new_tokens = 128) 
```




    [{'generated_text': 'ABSTRACT:Land use land cover (LULC) has altered dramatically because of anthropogenic activities, particularly in places where climate change and population growth are severe. The geographic information system (GIS) and remote sensing are widely used techniques for monitoring LULC changes. This study aimed to assess the LULC changes and predict future trends in Selangor, Malaysia. The satellite images from 1991–2021 were classified to develop LULC maps using support vector machine (SVM) classification in ArcGIS. The image classification was based on six different LULC classes, i.e., (i) water, (ii) developed, (iii) barren, (iv) forest, (v) agriculture, and (vi) wetlands. The resulting LULC maps illustrated the area changes from 1991 to 2021 in different classes, where developed, barren, and water lands increased by 15.54%, 1.95%, and 0.53%, respectively. However, agricultural, forest, and wetlands decreased by 3.07%, 14.01%, and 0.94%, respectively. The cellular automata-artificial neural network (CA-ANN) technique was used to predict the LULC changes from 2031–2051. The percentage of correctness for the simulation was 82.43%, and overall kappa value was 0.72. The prediction maps from 2031–2051 illustrated decreasing trends in (i) agricultural by 3.73%, (ii) forest by 1.09%, (iii) barren by 0.21%, (iv) wetlands by 0.06%, and (v) water by 0.04% and increasing trends in (vi) developed by 5.12%. The outcomes of this study provide crucial knowledge that may help in developing future sustainable planning and management, as well as assist authorities in making informed decisions to improve environmental and ecological conditions. ( aeros of. the, and. the, apology there identified. of thee and. data the interpretatione'}]


