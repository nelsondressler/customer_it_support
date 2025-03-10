# Take-Home Assignment: Email Classification

## Step 1: Data Collection and Preparation

Description

Gather a dataset of emails for training and testing purposes:
- Collect a diverse set of emails that represent different categories
and priorities. Ensure the dataset is balanced to avoid bias.

### Use Case

**Dataset Details**
- Source: Kaggle
- Author: Tobias Bueck
- Name: Customer IT Support - Ticket Dataset
- Description: Labeled Email Tickets with Agents answer, priorities, queues
- Link: [multilingual-customer-support-tickets](https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets)

**Features / Attributes**
- Queue: Specifies the department to which the email ticket is routed. The values can be:
    - Technical Support: Technical issues and support requests.
    - Customer Service: Customer inquiries and service requests.
    - Billing and Payments: Billing issues and payment processing.
    - Product Support: Support for product-related issues.
    - IT Support: Internal IT support and infrastructure issues.
    - Returns and Exchanges: Product returns and exchanges.
    - Sales and Pre-Sales: Sales inquiries and pre-sales questions.
    - Human Resources: Employee inquiries and HR-related issues.
    - Service Outages and Maintenance: Service interruptions and maintenance.
    - General Inquiry: General inquiries and information requests.
- Priority:	Indicates the urgency and importance of the issue. The values can be:
    - 1 (Low): Non-urgent issues that do not require immediate attention. Examples: general inquiries, minor inconveniences, routine updates, and feature requests.
    - 2 (Medium): Moderately urgent issues that need timely resolution but are not critical. Examples: performance issues, intermittent errors, and detailed user questions.
    - 3 (Critical): Urgent issues that require immediate attention and quick resolution. Examples: system outages, security breaches, data loss, and major malfunctions.
- Language:	Indicates the language in which the email is written. Useful for language-specific NLP models and multilingual support analysis. The values can be:
    - en (English).
    - de (German).
- Subject: Subject of the customer's email.
- Body: Body of the customer's email.
- Answer: The response provided by the helpdesk agent, containing the resolution or further instructions. Useful for analyzing the quality and effectiveness of the support provided.
- Type: Different types of tickets categorized to understand the nature of the requests or issues. The values can be:
    - Incident: Unexpected issue requiring immediate attention.
    - Request: Routine inquiry or service request.
    - Problem: Underlying issue causing multiple incidents.
    - Change: Planned change or update.
- Business Type: The business type of the support helpdesk. Helps in understanding the context of the support provided. Examples: "Tech Online Store", "IT Services", "Software Development Company".
- Tags: Tags/categories assigned to the ticket to further classify and identify common issues or topics. Examples: "Product Support", "Technical Support", "Sales Inquiry".

**Ticketsystem — Ticket Flow**

<center><img src='https://miro.medium.com/v2/resize:fit:720/format:webp/1*7Swk3VoxoALHTfWetzvgLg.png'/></center>

**Network Diagram Tags**

<center><img src='https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F3023333%2F9f9df25b75671db2d255b2d284c2c80c%2Fnetwork_diagram.svg?generation=1739380045025331&alt=media' width='800'/></center>

**Project Details**

Context: It's common in companies to have customer support systems that have to deal with emails from customers which include tasks such as topic comprehension and criticality understanding in order to refer to an ideal department and prioratize according to level of urgency.

Problem: How to optimize and improve customer support systems in order to classify automatically emails by the respective topic and priority labels?

Solution: Create a specialist AI model based on Generative AI to perform a text classification and assign the correct label for each email regarding to both topic and priority.

## Step 2: Model Training and Fine-Tuning

Description

Fine-tune the selected GenAI model on the prepared email dataset

## Step 3: Chatbot Development

Develop a simple and intuitive UI that allows users to input emails and
receive categorized outputs.

Implement a chatbot that integrates with the trained GenAI model:

## Step 4: Evaluation and Deployment

Evaluate the model's performance on test data

Approach:
- Exploratory Data Analysis
- Text Preprocessing
- Text Embedding
- Machine Learning (Base Line)
- Deep Learning / Transformers / Generative AI (Model Selection)
- Fine Tuning
- Model Evaluation

Technologies and Tools:
- Python
    - OOP
    - Poetry
    - Numpy
    - Pandas
    - Matplotlib/Seaborn
    - NLTK/Spacy
    - Scikit-Learn
    - Tensorflow/Keras/Pytorch/Transformers
    - Gradio/Streamlit
    - FastAPI
- Kaggle/HuggingFace
    - Customer IT Support - Ticket Dataset
    - BERT/DistilBERT
- Weights&Biases
- Git
- Docker

Project Instructions:
Requisites:
- Python installed (version 3.11.6)
- Poetry installed (version 2.1.1)
To run the code:
- Download the project or clone directly from the repo on GitHub
- Open the terminal on your OS
- Navigate up to the project root folder (customer_it_support)
- Run the script according to your OS:
    - In case of Unix-based, run:
        ```bash
        ./run_all_sh
        ```

    - In case of Windows, run:
        ```bat
        run_all.bat
        ```

- Open three tabs on your preferred browser and on each one enter the respective URL:
    - API: http://0.0.0.0:8000/
    - Gradio_App: http://127.0.0.1:7860/
    - Streamlit_App: http://localhost:8501/

TODO List:
- Split the code of API and UIs
- Add an endpoint to train models
- Add more types of models not only of baselines but also of transformers
- Add a feature to upload datasets to the UI and evaluate chunks of data
- Deploy the Gradio app to HuggingFace space
- Deploy the Streamlit app to the platform
- Try to get advantage of tranformers to improve the accuracy of classifying the textual data
- Add unit test for each class method or entire data flow
- Add integration tests between apps and API
- Use a tool or a cloud based solution to perform an entire CI/CD flow

## References

- Customer IT Support - Ticket Dataset:
    - [Kaggle](https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets?resource=download)
    - [Huggingface](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets)

- [Email Ticket Text Classification Dataset](https://medium.com/@softoft/email-ticket-text-german-classification-dataset-772d345e7a10)


- [Generative AI for Text Classification](https://medium.com/@memrhimanshu/generative-ai-for-text-classification-1ceee4a0da79)


- [BERT-Base](https://huggingface.co/docs/transformers/en/model_doc/bert)

- [DistilBERT](https://huggingface.co/docs/transformers/en/model_doc/distilbert)

- [Building an Email Classification Model with HuggingFace](https://balkaranbrar.medium.com/building-an-email-classification-model-with-huggingface-5e5e7f8f93b7)

- [Fine-Tuning BERT for Text Classification: A Step-by-Step Guide with Code Examples](https://medium.com/@somasunder/fine-tuning-bert-for-text-classification-a-step-by-step-guide-with-code-examples-0dea8513bcf2)

- [Fine tune BERT for text classification](https://medium.com/codex/fine-tune-bert-for-text-classification-cef7a1d6cdf1)

- [Fine tuning a custom model with Gemini](https://medium.com/@sulbha.jindal/fine-tuning-a-custom-model-with-gemini-f94809086bf6)

- [BERT Text Classification using Keras](https://swatimeena989.medium.com/bert-text-classification-using-keras-903671e0207d)

- [Optimize Hugging Face models with Weights & Biases](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/huggingface/Optimize_Hugging_Face_models_with_Weights_%26_Biases.ipynb#scrollTo=8XitzzgZN65d)

- [Machine Learning Model deployment with FastAPI, Streamlit and Docker](https://medium.com/latinxinai/fastapi-and-streamlit-app-with-docker-compose-e4d18d78d61d)

- [Python Poetry – The Best Data Science Dependency Management Tool?](https://towardsdatascience.com/python-poetry-the-best-data-science-dependency-management-tool-cca260257dd5/)