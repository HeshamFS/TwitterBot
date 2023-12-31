import os
from dotenv import load_dotenv

from langchain import PromptTemplate, LLMChain
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
from bs4 import BeautifulSoup
import requests
import json
from langchain.schema import SystemMessage
from twit import tweeter
from fastapi import FastAPI


load_dotenv()
brwoserless_api_key = os.getenv("BROWSERLESS_API_KEY")
serper_api_key = os.getenv("SERP_API_KEY")

# 1. Tool for search


def search(query):
    url = "https://google.serper.dev/search"

    payload = json.dumps({
        "q": query
    })

    headers = {
        'X-API-KEY': serper_api_key,
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)

    return response.text


# 2. Tool for scraping
def scrape_website(objective: str, url: str):
    # scrape website, and also will summarize the content based on objective if the content is too large
    # objective is the original objective & task that user give to the agent, url is the url of the website to be scraped

    print("Scraping website...")
    # Define the headers for the request
    headers = {
        'Cache-Control': 'no-cache',
        'Content-Type': 'application/json',
    }

    # Define the data to be sent in the request
    data = {
        "url": url
    }

    # Convert Python object to JSON string
    data_json = json.dumps(data)

    # Send the POST request
    post_url = f"https://chrome.browserless.io/content?token={brwoserless_api_key}"
    response = requests.post(post_url, headers=headers, data=data_json)
    
    # Check the response status code
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text()
        print("CONTENTTTTTT:", text)

        if len(text) > 10000:
            output = summary(objective, text)
            
            return output
        else:
            return text
    else:
        print(f"HTTP request failed with status code {response.status_code}")



def summary(objective, content):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500)
    docs = text_splitter.create_documents([content])
    map_prompt = """
    Extract the key information for the following text for {objective}. The text is Scraped data from a website so 
    will have a lot of usless information that doesnt relate to this topic, links, other news stories etc.. 
    Only summarise the relevant Info and try to keep as much factual information Intact
    Do not describe what the webpage is, you are here to get acurate and specific information
    Example of what NOT to do: "Investor's Business Daily: Investor's Business Daily provides news and trends on AI stocks and artificial intelligence. They cover the latest updates on AI stocks and the trends in artificial intelligence. You can stay updated on AI stocks and trends at [AI News: Artificial Intelligence Trends And Top AI Stocks To Watch "
    Here is the text:

    "{text}"
    SUMMARY:
    """
    map_prompt_template = PromptTemplate(
        template=map_prompt, input_variables=["text", "objective"])

    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type='map_reduce',
        map_prompt=map_prompt_template,
        combine_prompt=map_prompt_template,
        verbose=True
    )

    output = summary_chain.run(input_documents=docs, objective=objective)

    return output

class ScrapeWebsiteInput(BaseModel):
    """Inputs for scrape_website"""
    objective: str = Field(
        description="The objective & task that users give to the agent")
    url: str = Field(description="The url of the website to be scraped")


class ScrapeWebsiteTool(BaseTool):
    name = "scrape_website"
    description = "useful when you need to get data from a website url, passing both url and objective to the function; DO NOT make up any url, the url should only be from the search results"
    args_schema: Type[BaseModel] = ScrapeWebsiteInput

    def _run(self, objective: str, url: str):
        return scrape_website(objective, url)

    def _arun(self, url: str):
        raise NotImplementedError("error here")


# 3. Create langchain agent with the tools above
tools = [
    Tool(
        name="Search",
        func=search,
        description="useful for when you need to answer questions about current events, data. You should ask targeted questions"
    ),
    ScrapeWebsiteTool(),
]

system_message = SystemMessage(
content="""
You are an expert researcher with a keen ability to explore, analyze, and present information in a comprehensive manner. Your sources of information are diverse, credible, and well-informed. As you conduct this research, adhere to the following guidelines:

1/ Dedicate your efforts to gather extensive information concerning the Palestinian cause, emphasizing sources that are in support of the Palestinian narrative.
2/ Focus on factual data and events that underscore the history, struggles, and challenges faced by the Palestinian people in the last century.
3/ If reference links or articles are mentioned, extract relevant details to enhance the depth and context of the information.
4/ After collecting the initial information, reflect on whether additional searches could further enrich the content. However, don't go beyond three iterations of this process.
5/ Avoid conjecture and speculation. Ensure every statement is backed by verifiable facts, data, or credible accounts.
6/ Cite references and provide links where readers can delve further into the original sources of information.
7/ Aim for thoroughness in your response, ensuring it encompasses at least 500 words.
8/ Offer specifics, avoiding vague or general summaries. Each reference should come with specific details rather than broad overviews.

For clarity, avoid responses similar to:

1/ WIRED - The magazine covers a range of tech topics, including AI. Source: WIRED
2/ Middle East News - This portal offers updates from the Middle East region. Source: Middle East News

Instead, target detailed extracts and explicit data that offer readers a comprehensive understanding of the Palestinian cause from the pro-Palestinian perspective.
"""
)

agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    "system_message": system_message,
}

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")
memory = ConversationSummaryBufferMemory(
    memory_key="memory", return_messages=True, llm=llm, max_token_limit=1000)

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs=agent_kwargs,
    memory=memory,
)

from langchain.retrievers.multi_query import MultiQueryRetriever
template = """
You're a seasoned ghostwriter with expertise in crafting engaging Twitter threads. Using the provided information and your existing knowledge, compose an informative and engrossing Twitter thread on the given topic. Your first tweet should hook the reader to entice them to continue reading.

Guidelines for crafting the thread:
1. **Voice & Tone**:
   - Be *Informative and Clear*: Ensure clarity and precision. Use authoritative phrases like "Research indicates" or "Experts suggest."
   - Be *Casual and Engaging*: Use a conversational tone. Engage readers by posing occasional questions.
2. **Mood**:
   - *Educational*: Offer valuable insights or teach the reader something new.
   - *Inviting*: Encourage further exploration or engagement.
3. **Sentence Structure**:
   - *Varied Lengths*: Mix short points for emphasis with longer explanatory sentences.
   - *Descriptive Sentences*: Describe information rather than giving directives. For instance, "Opting for this method can lead to..."
4. **Transition Style**:
   - Be *Sequential and Logical*: Provide information in a clear, logical order.
   - *Visual Emojis*: Use emojis as visual markers or emphasis.
5. **Rhythm & Pacing**:
   - Ensure a *Steady Flow* from one point to the next.
   - Intersperse with *Data and Sources*: Bolster claims with statistics or expert opinions. Offer links for readers seeking a deeper dive.
6. **Signature Styles**:
   - *Intriguing Introductions*: Begin tweets or threads with a captivating hook.
   - *Question-Clarification Format*: Pose a broad question and follow with clarifying details.
   - *Engaging Summaries*: Recap concisely or invite further discussion.
7. **Distinctive Indicators**:
   - *Lead with Facts and Data*: Ground your content in research.
   - *Engaging Elements*: Use clear, descriptive sentences and occasional questions.
   - *Visual Emojis as Indicators*: Use emojis to mark transitions or emphasize points.
   - *Open-ended Conclusions*: Foster a sense of community by ending with questions or prompts.

Format Specifications:
- The thread should range between 3 and 10 tweets.
- Start each tweet with a format like (1/9), but omit the number for the first tweet.
- Use hashtags sparingly: no more than one or two for the entire thread.
- Only include relevant links without surrounding them with brackets.
- Ensure each tweet remains under 220 characters.
- Present each tweet as a separate paragraph.
- Use many relevant hashtags, includes: Gaza, Palestine, PalestineGenocide, PalestineLivesMatter, GazzeUnderAttack, FreePalestine, FreeGaza, Gazagenocide, SupportGaza
    Topic Headline:{topic}
    Info: {info}
    """

prompt = PromptTemplate(
    input_variables=["info","topic"], template=template
)

llm = ChatOpenAI(model_name="gpt-4")
llm_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    
)



  
twitapi = tweeter()

def tweetertweet(thread):

    tweets = thread.split("\n\n")
   
    #check each tweet is under 280 chars
    for i in range(len(tweets)):
        if len(tweets[i]) > 280:
            prompt = f"Shorten this tweet to be under 280 characters: {tweets[i]}"
            tweets[i] = llm.predict(prompt)[:280]
    #give some spacing between sentances
    tweets = [s.replace('. ', '.\n\n') for s in tweets]

    for tweet in tweets:
        tweet = tweet.replace('**', '')

    try:
        response = twitapi.create_tweet(text=tweets[0])
        id = response.data['id']
        tweets.pop(0)
        for i in tweets:
            print("tweeting: " + i)
            reptweet = twitapi.create_tweet(text=i, 
                                    in_reply_to_tweet_id=id, 
                                    )
            id = reptweet.data['id']
        return "Tweets posted successfully"
    except Exception as e:
        return f"Error posting tweets: {e}"


class TwitterMentionBot:
    def __init__(self):
        self.twitter_api = tweeter()
        self.twitter_me_id = self.get_me_id()
        self.tweet_response_limit = 5  # Limit the number of mentions we respond to at once

    def get_me_id(self):
        return self.twitter_api.get_me()[0].id

    def get_mentions(self):
        mentions = self.twitter_api.mentions_timeline(count=20)  # Retrieve the last 20 mentions
        for mention in mentions:
            print(f"Received mention from {mention.user.screen_name} with text: {mention.text}")
        return mentions


    def respond_to_mention(self, mention):
        # Extract the content of the mention
        mention_content = mention.text
        # Use the researchAgent function to research the topic and generate the response
        response_data = researchAgent(Query(query=mention_content))
        # Extract the tweet content from the response_data
        response_text = response_data['tweet']
        # Post the reply in response to the mention
        self.twitter_api.update_status(status=response_text, in_reply_to_status_id=mention.id)
        print(f"Replied to mention from {mention.user.screen_name} with text: {response_text}")
    
    def respond_to_mentions(self):
        try:
            mentions = self.get_mentions()
            for mention in mentions[:self.tweet_response_limit]:
                self.respond_to_mention(mention)
        except Exception as e:
            print(f"Error encountered: {e}")

# 5. Set this as an API endpoint via FastAPI
app = FastAPI()


class Query(BaseModel):
    query: str


@app.post("/")
def researchAgent( query: Query):
    query = query.query
    content = agent({"input": query})
    actual_content = content['output']
    thread = llm_chain.predict(info = actual_content, topic = query)
    ret = tweetertweet(thread)
    return ret

# Additional imports from the provided snippet
import tweepy
from datetime import datetime, timedelta
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# We already have dotenv imported, so we won't import it again.

# Using the tweeter function from twit.py to get the Tweepy client
client = tweeter()
print(f"Authenticated as: {client.get_me().screen_name}")


# class TwitterMentionBot:
#     def __init__(self):
#         self.twitter_api = client
#         self.twitter_me_id = self.get_me_id()
#         self.tweet_response_limit = 35

#         # Initialize the language model with temperature of .5 for some creativity
#         self.llm = ChatOpenAI(temperature=.5, model_name='gpt-4')


#     def get_me_id(self):
#         return self.twitter_api.get_me()[0].id

#     def get_mention_conversation_tweet(self, mention):
#         if mention.conversation_id is not None:
#             conversation_tweet = self.twitter_api.get_tweet(mention.conversation_id).data
#             return conversation_tweet
#         return None

#     def get_mentions(self):
#         now = datetime.utcnow()
#         start_time = now - timedelta(minutes=20)
#         start_time_str = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
#         return self.twitter_api.get_users_mentions(id=self.twitter_me_id,
#                                                    start_time=start_time_str,
#                                                    expansions=['referenced_tweets.id'],
#                                                    tweet_fields=['created_at', 'conversation_id']).data

    
    
#     def respond_to_mention(self, mention, mentioned_conversation_tweet):
#         # Extract the content of the original thread
#         original_thread_content = mentioned_conversation_tweet.text
        
#         # Use the researchAgent function to research the topic and generate the response
#         response_data = researchAgent(Query(query=original_thread_content))
        
#         # Extract the tweet content from the response_data
#         response_text = response_data['tweet']
        
#         # Post the reply in response to the mention
#         try:
#             self.twitter_api.create_tweet(text=response_text, in_reply_to_tweet_id=mention.id)
#         except Exception as e:
#             print(e)
#             return

#         # Step 1: Extract the content of the original thread
#         original_thread_content = mentioned_conversation_tweet.text
        
#         # Step 2: Use the search function to research the topic mentioned in the original thread
#         research_results = search(original_thread_content)
        
#         # Step 3: Generate a reply based on the results of the research
#         # We'll use the generate_response function to craft the reply
#         response_text = self.generate_response(research_results)
        
#         # Step 4: Post the reply in response to the mention
#         try:
#             self.twitter_api.create_tweet(text=response_text, in_reply_to_tweet_id=mention.id)
#         except Exception as e:
#             print(e)
#             return

#         response_text = self.generate_response(mentioned_conversation_tweet.text)
#         try:
#             self.twitter_api.create_tweet(text=response_text, in_reply_to_tweet_id=mention.id)
#         except Exception as e:
#             print(e)
#             return

#     def respond_to_mentions(self):
#         mentions = self.get_mentions()
#         if not mentions:
#             print("No mentions found")
#             return
#         for mention in mentions[:self.tweet_response_limit]:
#             mentioned_conversation_tweet = self.get_mention_conversation_tweet(mention)
#             if mentioned_conversation_tweet.id != mention.id:
#                 self.respond_to_mention(mention, mentioned_conversation_tweet)

# End of the extended content


@app.post("/respond")
def respond_to_query_or_mention(context: str, query: str = None):
    # If the context is a direct query, generate a series of tweets
    if context == "direct_query":
        content = agent({"input": query})
        actual_content = content['output']
        thread = llm_chain.predict(info=actual_content, topic=query)
        return tweetertweet(thread)
    # If the context is a mention, generate a single tweet in reply
    elif context == "mention":
        # mention_bot = TwitterMentionBot()
        # mention_bot.respond_to_mentions()
        mention_bot = TwitterMentionBot()
        mention_bot.respond_to_mentions()

        return "Replied to mentions"
    else:
        return "Invalid context"

