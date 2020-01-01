class Foo:
    def __call__(self, i=6):
        print('data={0}'.format(i))


foo_instance = Foo()
foo_instance()  # this is calling the __call__ method
