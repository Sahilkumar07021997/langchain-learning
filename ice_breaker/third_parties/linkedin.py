import os
import requests
from dotenv import load_dotenv

load_dotenv()


def scrape_linkedin_profile(linkedin_profile_url: str, mock: bool = False):
    """Scrape the information from LinkedIn profiles,
    manually scrape the information from LinkedIn profile"""

    if mock:
        linkedin_profile_url = "https://gist.githubusercontent.com/Sahilkumar07021997/b1850e39b9ea7cfb554cc3e0418ca446/raw/a353ed83760ffb9fb8f3285313a630520e730454/sahil-kumar-aa868218b.json"
        response = requests.get(url=linkedin_profile_url,
                                timeout=10,
                                )
    else:
        api_key = os.environ.get("PROXY_CURL_API_KEY")
        headers = {'Authorization': 'Bearer ' + api_key}
        api_endpoint = 'https://nubela.co/proxycurl/api/v2/linkedin'
        params = {
            # 'twitter_profile_url': 'https://x.com/johnrmarty/',
            # 'facebook_profile_url': 'https://facebook.com/johnrmarty/',
            'linkedin_profile_url': 'https://www.linkedin.com/in/sahil-kumar-aa868218b',
            # 'extra': 'include',
            # 'github_profile_id': 'include',
            # 'facebook_profile_id': 'include',
            # 'twitter_profile_id': 'include',
            # 'personal_contact_number': 'include',
            # 'personal_email': 'include',
            # 'inferred_salary': 'include',
            # 'skills': 'include',
            # 'use_cache': 'if-present',
            # 'fallback_to_cache': 'on-error',
        }
        response = requests.get(api_endpoint,
                                params=params,
                                headers=headers,
                                timeout=10)

    data = response.json()

    return data


if __name__ == "__main__":
    print(scrape_linkedin_profile(linkedin_profile_url="https://www.linkedin.com/in/sahil-kumar-aa868218b",
                                  mock=True))
