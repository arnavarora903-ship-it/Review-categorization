# import streamlit as st
# from langchain import PromptTemplate
# from langchain_openai import OpenAI


# template = """\
# For the following text, extract the following \
# information:

# sentiment: Is the customer happy with the product? 
# Answer Positive if yes, Negative if \
# not, Neutral if either of them, or Unknown if unknown.

# delivery_days: How many days did it take \
# for the product to arrive? If this \
# information is not found, output No information about this.

# price_perception: How does it feel the customer about the price? 
# Answer Expensive if the customer feels the product is expensive, 
# Cheap if the customer feels the product is cheap,
# not, Neutral if either of them, or Unknown if unknown.

# Format the output as bullet-points text with the \
# following keys:
# - Sentiment
# - How long took it to deliver?
# - How was the price perceived?

# Input example:
# This dress is pretty amazing. It arrived in two days, just in time for my wife's anniversary present. It is cheaper than the other dresses out there, but I think it is worth it for the extra features.

# Output example:
# - Sentiment: Positive
# - How long took it to deliver? 2 days
# - How was the price perceived? Cheap

# text: {review}
# """

# #PromptTemplate variables definition
# prompt = PromptTemplate(
#     input_variables=["review"],
#     template=template,
# )


# #LLM and key loading function
# def load_LLM(openai_api_key):
#     """Logic for loading the chain you want to use should go here."""
#     # Make sure your openai_api_key is set as an environment variable
#     llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
#     return llm


# #Page title and header
# st.set_page_config(page_title="Extract Key Information from Product Reviews")
# st.header("Extract Key Information from Product Reviews")


# #Intro: instructions
# col1, col2 = st.columns(2)

# with col1:
#     st.markdown("Extract key information from a product review.")
#     st.markdown("""
#         - Sentiment
#         - How long took it to deliver?
#         - How was its price perceived?
#         """)

# with col2:
#     st.write("Contact with [AI Accelera](https://aiaccelera.com) to build your AI Projects")


# #Input OpenAI API Key
# st.markdown("## Enter Your OpenAI API Key")

# def get_openai_api_key():
#     input_text = st.text_input(label="OpenAI API Key ",  placeholder="Ex: sk-2twmA8tfCb8un4...", key="openai_api_key_input", type="password")
#     return input_text

# openai_api_key = get_openai_api_key()


# # Input
# st.markdown("## Enter the product review")

# def get_review():
#     review_text = st.text_area(label="Product Review", label_visibility='collapsed', placeholder="Your Product Review...", key="review_input")
#     return review_text

# review_input = get_review()

# if len(review_input.split(" ")) > 700:
#     st.write("Please enter a shorter product review. The maximum length is 700 words.")
#     st.stop()

    
# # Output
# st.markdown("### Key Data Extracted:")

# if review_input:
#     if not openai_api_key:
#         st.warning('Please insert OpenAI API Key. \
#             Instructions [here](https://help.openai.com/en/articles/4936850-where-do-i-find-my-secret-api-key)', 
#             icon="⚠️")
#         st.stop()

#     llm = load_LLM(openai_api_key=openai_api_key)

#     prompt_with_review = prompt.format(
#         review=review_input
#     )

#     key_data_extraction = llm(prompt_with_review)

#     st.write(key_data_extraction)


# OUR OWN VERSION OF THIS APPLICATION: LET'S GO 

from enum import Enum 
import streamlit as st 
from typing import List 
from pydantic import BaseModel,Field 
from langchain_openai import ChatOpenAI 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate 

# from langchain_core.output_parsers import StrOutputParser

# Output parser are designed for raw text outputs. 
# Using any output parser after .with_structured_output() makes no sense 
# as the output is already in good structured format and validated. Thus, applying 
# any output parser like stroutputparser on it turns it into ambiguous text, this makes redcuing less 
# reliable.

# Schema Code 

# use same case everywhere either snake_case or Pascal_Case. Do not mix them.
# use multi line descriptions it makes code more cleaner. Further, it makes easier to understand and use
# never have any spelling error in any part of code especially in shcema code 

class SentimentLabel(str,Enum):
    positive='positive' # no trailing commans when defining Enum values 
    negative='negative'
    neutral='neutral'
    mixed='mixed'
class GroundingStatus(str,Enum):
    supported='supported' # no trailing commas here in Enum values 
    partially_supported = 'partially_supported'
    not_supported='not_supported'

class GroundCheck(BaseModel):
    status:GroundingStatus  = Field(
        description='''Whether the review is backed by review text
        Any one of this: Supported,Partially-Supported,Not Supported'''
    )
    explanation:str = Field(
        description='''One line reasoning explaining the status choice.'''
    )

# descriptions are not comments they directly influence llm behaviour and UI rendering 
# Enums are for control if a value must be restricted then never use str 
# Use .with_structured_output() with llm initializing code 
# llm cannot interpret python classes so pass them using .with_structured_output()
# do not make LLM, PROMPT and CHAIN in for loop. Make them outside for loop 
# never pass schema objects in prompts 
# Mapping must be cheap, strict. No creativity, no reasoning 
# Python is case sensitive, it uses None (capital N)
# PROMPT WORDING MATTERS 

class OverallSentimentLabel(BaseModel): # overall sentiment label has many features in it thus it makes sense to create a separate class for it 
    label:SentimentLabel = Field (
        description='''Overall view of the review.'''
    )
    reasoning: str = Field(
        description='''Explaining point of view in 1 to 2 lines.'''
    )
    grounding_check: GroundCheck = Field(
        description='''Verification of the overall sentiment.'''
    )

class MappingBusinessInsights(BaseModel): # all the schemas in this class do not have many features in them thus, it makes sense to include them in a single class 
    overall_sentiment:OverallSentimentLabel = Field(
        description=''' This gives overall vide of review along with reasoning .'''
    )

    positive_aspects: list[str] = Field(
        description='''All the positive qualities that are mentioned in the review.
         Use ["not mentioned"] if nothing is mentioned'''
    )
    negative_aspects: list[str] = Field(
        description='''Negative qualities mentioned about the product.
         Use ["not mentioned"] if nothing is mentioned'''
    )
    quality_or_durability: list[str] = Field(
        description='''Key details shared regarding quality or durability of the product.
         Use ["not mentioned"] if nothing is mentioned'''
    )
    price:list[str] = Field(
        description='''Customers statements about price or value
        Use ["not mentioned"] if nothing is mentioned'''
    )
    shipping_experience: list[str] = Field(
        description='''Comments by customers which mention their experience
        with delivery, packaging, speed etc.
        Use ["not mentioned"] if nothing is mentioned.'''
    )
    reason_for_purchasing:list[str] = Field(
        description='''This points out to why,how or for whom customer
        purchased the product.
        Use ["not mentioned"] if nothing is mentioned.'''
    )
    customer_suggestions:list[str]= Field(
        description='''This states scopes of improvements brought forward by customer
        Use ["not mentioned"] if nothing is mentioned.'''
    )

def mapping_llm(api_key): # reading all chunks to generate parital answers does not require much creativity 
    llm = ChatOpenAI(api_key=api_key,model='gpt-4.1-nano',temperature=0)
    return llm
def reducing_llm(api_key): # combining all parital answers into one whole answer is creative work
    llm = ChatOpenAI(api_key=api_key,model='gpt-4.1-nano',temperature=0.2)
    return llm

def get_openai_api_key():
    api_key=st.sidebar.text_input(placeholder='sk-',label='Your key')
    return api_key 

api_key = get_openai_api_key()

st.set_page_config(page_title="Review's review")
st.markdown('Hey! I am here to help you')
st.markdown('**Paste review here**')

review=st.text_area(placeholder='review',label='review_area',label_visibility='collapsed')

text_splitter = RecursiveCharacterTextSplitter(
    separators=['.','\n'],# here I want to separte first with a '.' and then with a '\n'
    chunk_size=150,
    chunk_overlap=50
)
if review.strip(): # .strip () ensures that review is just not empty or spaces but has content 
    splitted_review=text_splitter.create_documents([review])# this is a list of chunks of review
    if splitted_review:
        partial_answers=[]
        prompt = ChatPromptTemplate.from_messages([
                ('system',''' Extract required information,do not hallucinate. '''),
                ('human','''{chunk}''')
            ])
        llm = mapping_llm(api_key).with_structured_output(MappingBusinessInsights)
        chain = prompt|llm 

        for doc in splitted_review:
            # doc is a document. you cannot pass it directly to llm.
            # pass the page content of the document 
            try: # this ensures that even if there is any problem with even any of the chunks the code will still continue 
                response =chain. invoke({'chunk':doc.page_content})
                partial_answers.append(response)
            except Exception:
                continue
    else:
        st.stop()
else: 
    st.stop()

# now we have a list of all the partial answers from llm. Mapping is complete 
# today we completed mapping, tomorrow we will continue with reducing and testing. 

# lets begin reducing and then testing 

# step 1: combining all the objects from parital answers into 1 single object 

# from collections import defaultdict # this will help to automatically create a dictionary of lists 
# from typing import get_origin # this will help in identifying what is the origin of the a field ex list[str] here get_origin will give list (that is the base type)


# def aggregate_partial_answers(partial_answers): # this function takes partial_answers as input and gives a single aggregated object which is a dictionary as output 
#     aggregated={}
#     lists = defaultdict(list) # this automatically creates empty list for any new key 

#     for field,info in BusinessInsights.model_fields.items(): # model_fields is a pydantic method that gives all the fields of a class as a dict then .items() gives key value pairs Which later python unpacks as field and info 
#         field_type = info.annotation 

#         if get_origin(field_type) is list:
#             for obj in partial_answers:
#                 lists[field].extend(getattr(obj,field,[]))
            
#             cleaned = [v for v in lists[field] if v.lower() != "not mentioned"]
#             aggregated[field]=cleaned or ["not mentioned"]

#         elif isinstance (field_type,type) and issubclass(field_type,BaseModel):
#             aggregated[field]=[getattr (obj,field) for obj in partial_answers]
    
#     return aggregated

# AGGREGATION CODE 
from collections import defaultdict 
from typing import get_origin 

def aggregate_partial_answers(partial_answers):
    aggregated = {} # cleaned and final which will be passed to reducer llm 
    lists = defaultdict(list) # this is the unclean one it has redundancy's in it 

    for field, info in MappingBusinessInsights.model_fields.items():
        field_type=info.annotation # this info.annototation is a command in model_fields. This gives the type of the field 
        if get_origin(field_type) is list:
            for obj in partial_answers:
                lists[field].extend(getattr(obj,field,[])) # unclean
                cleaned=[v for v in lists[field] if v.lower() != "not mentioned"]
                aggregated[field] = cleaned or ["not mentioned"]
        elif isinstance(field_type,type) and issubclass(field_type,BaseModel):
                aggregated[field]=[getattr(obj,field) for obj in partial_answers]
    return aggregated  

# REDUCER CODE 
reducer_prompt = ChatPromptTemplate.from_messages([
    ('system',"""You are a reducer that consolidates structured insights extracted from multiple text chunks.

You will receive aggregated fields where each field may contain multiple entries collected from different chunks.

Rules you MUST follow:

GENERAL
- Do not invent new information.
- Use only the provided content.
- Preserve meaning; do not over-abstract.
- Prefer wording close to the original phrases.
- Remove redundancy.

LIST FIELDS
- Merge entries only if they clearly express the same underlying idea.
- Do NOT merge loosely related or partially overlapping ideas.
- If similarity is uncertain, keep items separate.
- When merging, produce a single clear and concise entry.

SENTIMENT LABEL
- Choose exactly one: positive, negative, or mixed.
- Choose "mixed" ONLY if there is clear evidence of both positive and negative sentiment.
- If "mixed" is chosen, explicitly explain:
  - What evidence is positive
  - What evidence is negative
- Do not choose "mixed" if one side clearly dominates.

OUTPUT
- Follow the provided schema exactly.
- If no valid information exists for a field, output "not mentioned".
"""),
('human',"""Using the aggregated structured data below, produce a final coherent, clear, and concise structured output.

- Remove duplicates and redundant phrasing.
- Apply semantic similarity carefully according to the rules.
- Ensure internal consistency across all fields.

Aggregated data:
{aggregated_data}
""")
])

# REDUCER SCHEMA 

class SentimentLabel(str, Enum):
    positive = "positive"
    negative = "negative"
    mixed = "mixed"


class GroundCheck(BaseModel):
    status: str = Field(
        description=(
            "Whether the sentiment conclusion is supported by aggregated evidence. "
            "One of: Supported, Partially-Supported, Not Supported"
        )
    )
    explanation: str = Field(
        description="One-line explanation justifying the grounding status."
    )


class OverallSentiment(BaseModel):
    label: SentimentLabel = Field(
        description="Final overall sentiment derived from all aggregated evidence."
    )
    reasoning: str = Field(
        description=(
            "Mandatory explanation ONLY when label is 'mixed'. "
            "Must explicitly mention both positive and negative evidence. "
            "If label is positive or negative, use 'not mentioned'."
        )
    )
    grounding_check: GroundCheck = Field(
        description="Verification that the sentiment is grounded in review content."
    )


class ReducingBusinessInsights(BaseModel):
    overall_sentiment: OverallSentiment

    positive_aspects: List[str] = Field(
        description="All positive qualities mentioned. Use ['not mentioned'] if none."
    )
    negative_aspects: List[str] = Field(
        description="All negative qualities mentioned. Use ['not mentioned'] if none."
    )
    quality_or_durability: List[str] = Field(
        description="Statements about quality or durability. Use ['not mentioned'] if none."
    )
    price: List[str] = Field(
        description="Statements related to price or value. Use ['not mentioned'] if none."
    )
    shipping_experience: List[str] = Field(
        description="Comments on delivery, packaging, or shipping speed. Use ['not mentioned'] if none."
    )
    reason_for_purchasing: List[str] = Field(
        description="Why or for whom the product was purchased. Use ['not mentioned'] if none."
    )
    customer_suggestions: List[str] = Field(
        description="Improvement suggestions from the customer. Use ['not mentioned'] if none."
    )
# Now we have aggregated dictionary which has evidence to be passed to 
# to reducer llm, reducer prompt and reducer schema 
# So now next step should be to create the reducer chain 

reducer_llm = reducing_llm(api_key).with_structured_output(ReducingBusinessInsights) # reducer llm iniatilized 
reducer_chain= reducer_prompt|reducer_llm
aggregated_data=aggregate_partial_answers(partial_answers)

final_output=reducer_chain.invoke({'aggregated_data':aggregated_data})
st.markdown('### Final Extracted Insights')
st.json(final_output.model_dump())



# IT IS WORKING GREAT VERY VERY GOOD :) 
