import scrapy


ROOT_URL = 'https://www.thesaurus.com/browse/'
    

class ThesaurusSpider(scrapy.Spider):
    name = 'thesaurus_spider'
    start_urls = [ROOT_URL + 'good']

    def parse(self, response):
        word = response.css('h1.css-eg1f2v::text').get()

        synonyms = []
        for synonym in response.css('a.css-3kshty::text'):
            synonyms.append(synonym.get())
            yield {'w1': word, 'w2': synonym.get(), 'sign': 1}

        antonyms = []
        for antonym in response.css('a.css-1yg9g8p::text'):
            antonyms.append(antonym.get())
            yield {'w1': word, 'w2': antonym.get(), 'sign': -1}

        for w in (synonyms + antonyms):
            yield response.follow(ROOT_URL + w, self.parse)
