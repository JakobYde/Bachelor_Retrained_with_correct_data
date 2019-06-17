import csv
import numpy as np

class DataSheet:
    def __init__(self, filename=None):
        self.data = {}
        self.types = []
        self.categories = []

        if filename != None:
            self.load_file(filename)

    def set_head(self, row):
        assert (isinstance(row, list)),'Row should be a list.'
        for cell in row:
            self.data[cell] = []

    def get_head(self):
        return [key for key in self.data]

    def set_types(self, row):
        assert (isinstance(row, list) and (False not in [isinstance(cell, str) for cell in row])),'Row should be a list of strings.'
        self.types = []
        for cell in row:
            assert (cell in ['int', 'float', 'str']),'Invalid type in row.'
            
            if cell == 'int': self.types.append(int)
            if cell == 'str': self.types.append(str)
            if cell == 'float': self.types.append(float)

    def get_types(self):
        return self.types

    def set_categories(self, row):
        assert (isinstance(row, list) and (False not in [(cell in ['in', 'out']) for cell in row])),'Row should be a list of valid categories.'
        self.categories = []
        for cell in row:
            self.categories.append(cell)

    def get_categories(self):
        return self.categories

    def add_row(self, row):
        assert (isinstance(row, list)),'Row should be a list.'
        for i, key in enumerate(self.data):
            self.data[key].append(self.types[i](row[i]))

    def get_data(self, parameter=None):
        if parameter == None: return self.data
        else: return self.data[parameter]

    def load_file(self, filename):
        data = []
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                data.append(list(row))
        self.set_head(data[0])
        self.set_categories(data[1])
        self.set_types(data[2])

        for row in data[3:]:
            self.add_row(row)

    def include(self, parameters=None, categories=None, types=None):
        result = DataSheet()
        for i, key in enumerate(self.data):
            if parameters == None or key in parameters:
                if categories == None or self.categories[i] in categories:
                    if types == None or self.types[i] in types:
                        result.categories.append(self.categories[i])
                        result.types.append(self.types[i])
                        result.data[key] = self.data[key]
        return result

    def exclude(self, parameters=None, categories=None, types=None):
        result = DataSheet()
        for i, key in enumerate(self.data):
            if parameters == None or key not in parameters:
                if categories == None or self.categories[i] not in categories:
                    if types == None or self.types[i] not in types:
                        result.categories.append(self.categories[i])
                        result.types.append(self.types[i])
                        result.data[key] = self.data[key]
        return result

    def remove_by_performance(self, parameter, threshold=None):
        assert (parameter in self.data),'Invalid parameter.'
        n = 0
        if threshold == None:
            mae = np.mean(np.abs(self.data[parameter] - np.mean(self.data[parameter])))
            threshold = np.mean(self.data[parameter]) + mae
        elements = list(self.data[parameter])
        elements.append(threshold)
        elements.sort(reverse=True)
        elements = elements[:elements.index(threshold)]
        for element in elements:
            n += 1
            index = self.data[parameter].index(element)
            #print(element, index)
            for key in self.data:
                del self.data[key][index]
        m = max(self.data[parameter])
        return n