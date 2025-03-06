import wikipedia

from haystack_experimental.components.summarizer.summariser import Summarizer
from haystack import Document


def example():

    page = wikipedia.page("Rolling_Stones")
    textual_content = page.content
    doc = Document(content=textual_content)

    summariser = Summarizer()
    summariser.run(documents=[doc])


    # load encoding and check the length of dataset
    # encoding = tiktoken.encoding_for_model('gpt-4o-mini')
    # len(encoding.encode(artificial_intelligence_wikipedia_text))
    # summary_with_detail_0 = summarize(artificial_intelligence_wikipedia_text, detail=0, verbose=True)