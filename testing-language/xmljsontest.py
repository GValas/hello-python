import xmltodict

mydict = {
    'text': {
        '@color':'red',
            '@stroke':'2',
            '#text':'This is a test'
    }
}
print(xmltodict.unparse(mydict, pretty=True))