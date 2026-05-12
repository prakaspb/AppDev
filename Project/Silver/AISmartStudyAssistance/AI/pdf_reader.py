import pdfplumber

def read_pdf(path):
    text=''
    with pdfplumber.open(path) as pdf:
        for p in pdf.pages:
            #text+=p.extract_text()+'
            text+=p.extract_text()+''

    return text

#Wish to have unit team to test this function, but I have no time to do it. So I will just write a simple test case here.
if __name__ == '__main__':
    path = 'E:\Prakash\Project_Silver\AISmartStudyDevImpl2\AI_SmartStudyAssistance\material\physics_high_school.pdf'
    text = read_pdf(path)
    print(text)
    