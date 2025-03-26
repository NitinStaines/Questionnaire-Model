# 📄 Comprehensive Questionnaire System  

## 📌 Overview  
The Comprehensive Questionnaire System is an interactive, AI-powered tool designed to automate the generation of summaries and multiple-choice questions (MCQs) from articles. It allows users to input article URLs, extract key content through web scraping, generate concise summaries, and create relevant MCQs to enhance reading comprehension and knowledge retention.  

## 🚀 Features  
- **Automated Web Scraping**: Extracts content from article URLs using Selenium.  
- **AI-Powered Summarization**: Uses a fine-tuned BART model for summarization.  
- **MCQ Generation**: Leverages a fine-tuned T5 model with LoRa to create context-aware multiple-choice questions.  
- **Interactive UI**: Enables users to input URLs, view summaries, answer questions, and receive feedback.  
- **Scalability & Integration**: Designed for real-time processing with backend support in Express.js and Flask.  

## 🎯 Problem Statement  
Current methods of content summarization and question generation require manual effort, making it challenging to create structured assessments efficiently. This system aims to:  
- Automate content extraction, summarization, and MCQ generation.  
- Improve learning experiences by engaging users with AI-generated questions.  
- Provide an accessible and scalable solution for education and knowledge testing.  

## 🛠️ Tech Stack  

### Core Libraries & Frameworks  
- **Web Scraping**: Selenium (for dynamic content extraction).  
- **NLP Models**: Hugging Face Transformers (BART for summarization, T5 for MCQ generation).  

### Frontend & Backend  
- **Frontend**: HTML, CSS, JavaScript for an interactive UI.  
- **Backend**: Flask (NLP model integration), Express.js (API and data flow management).  

## 📊 Dataset  
- The SQuAD v2 (Stanford Question Answering Dataset v2.0) is a popular dataset for reading comprehension and question-answering (QA) tasks.
= Total Questions: 150,000+
- Each passage is paired with multiple questions, some of which have no answers.Total Questions: 150,000+
- Passages: Extracted from Wikipedia
= Each passage is paired with multiple questions, some of which have no answers.

## 🏗️ System Architecture  
- **Data Ingestion Layer**: Users provide URLs, and Selenium extracts article content.  
- **Processing Layer**:  
  - BART model generates summaries.  
  - T5 model generates MCQs with relevant distractors.  
- **User Interaction Layer**: Users input URLs, answer MCQs, and receive instant feedback.  
- **Backend Processing**: Flask and Express.js manage interactions and data handling.  


## 👥 Contributors  
- Mathesh D  
- Monish S  
- Nitin Staines  

Snapshots of an application with the model used at the backend:
![image](https://github.com/user-attachments/assets/4832c3ee-25a3-4ce0-a504-2fe1544ecc64)
![image](https://github.com/user-attachments/assets/7969b492-5011-432c-8d07-5a8051149aaa)
![image](https://github.com/user-attachments/assets/564e56d9-f296-4b17-994a-a3a1cbcd6e26)

Feel free to use the model!
