"""Parent class for all fetcher subclasses"""

class BaseFetcher(object):

    def __init__(self, email=None, api_key=None):
        self.email      = email
        self.api_key    = api_key

    def get(self, term):
        pass

    def fetch(self, term):
        pass