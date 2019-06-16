class DataSheet:
    def __init__(self, filename=None):
        self.data = {}
        self.types = []

        if filename != None:
            load_file(filename)

    def set_head(self, row):
        assert (isinstance(row, list)),'Row should be a list.'
        for cell in row:
            self.data[cell] = []

    def set_types(self, row):
        assert (isinstance(row, list) and (False not in [isinstance(cell, str) for cell in row])),'Row should be a list of strings.'
        self.types = []
        for cell in row:
            assert (cell in ['int', 'float', 'str']),'Invalid type in row.'
            
            if cell == 'int': self.types.append(int)
            if cell == 'str': self.types.append(str)
            if cell == 'float': self.types.append(float)

    def set_categories(self, row):
        assert (isinstance(row, list) and (False not in [(cell in ['in', 'out']) for cell in row])),'Row should be a list of valid categories.'
        self.categories = []
        for cell in row:
            self.categories.append(cell)

    def add_row(self, row):
        assert (isinstance(row, list)),'Row should be a list.'
        for i, key in enumerate(self.data):
            self.data[key].append(self.types[i](row[i]))

    def load_file(self, filename):
        file = []
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                file.append(list(row))
        set_head(file[0])
        set_categores(file[1])
        set_types(file[2])

        for row in file[3:]:
            add_row(row)