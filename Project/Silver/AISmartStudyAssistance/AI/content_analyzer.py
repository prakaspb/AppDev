from sklearn.feature_extraction.text import TfidfVectorizer
from ai.pdf_reader import read_pdf
class ContentAnalyzer:
    def extract_concepts(self,text):
        vec=TfidfVectorizer(stop_words='english',max_features=4)
        X=vec.fit_transform([text])
        return vec.get_feature_names_out()

#text = read_pdf("E:\Prakash\Project_Silver\AISmartStudyDevImpl2\AI_SmartStudyAssistance\material\physics_high_school.pdf")
#Unit testing 
if __name__ == '__main__':
    #text = read_pdf('E:\Prakash\Project_Silver\AISmartStudyDevImpl2\AI_SmartStudyAssistance\material\physics_high_school.pdf')
    #text = "The cat is on the roof. The dog is in the garden. The cat and the dog are friends."
    analyzer = ContentAnalyzer()
    concepts = analyzer.extract_concepts(text)
    print(concepts)