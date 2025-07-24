from langchain_community.tools import TavilySearchResults


def get_linkedin_profile_url_tavily(name: str):
    """Scrap and search for linkedIn profile page for the given {name}"""
    search = TavilySearchResults()
    res = search.run(f"{name}")
    return res
