# importing all the required modules
import PyPDF2

# creating an object
file = open('testing-pypdf/pdf.pdf', 'rb')

# creating a pdf reader object
pdf = PyPDF2.PdfFileReader(file)

# print the number of pages in pdf file
print(pdf.numPages)
