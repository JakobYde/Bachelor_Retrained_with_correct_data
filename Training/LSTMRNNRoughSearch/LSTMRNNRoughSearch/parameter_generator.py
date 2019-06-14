from random import choice, random
from random import seed as random_seed

class Parameter:
    def __init__(self, default_value = None, change_chance = None, choices = []):
        assert ((change_chance is None) or (0 <= change_chance <= 1)), 'Chance should be between 0 and 1 if specified.'
        assert (change_chance == None or (default_value != None and choices != []))
        assert isinstance(choices, list)

        if default_value != None:
            self.default_value = default_value
        else:
            self.change_chance = 1
            self.default_value = None

        if choices != []:
            self.choices = choices
        else:
            self.change_chance = 0
        
        if default_value != None and choices != []:
            self.change_chance = change_chance

        pass

    def set_default(self, new_default):
        self.default_value = new_default

    def set_choices(self, new_choices):
        assert isinstance(new_choices, list)
        self.choices = new_choices

    def set_chance(self, new_chance):
        assert (0 <= new_chance <= 1), 'Chance should be between 0 and 1.'
        self.change_chance = new_chance
        
    def get_permutations(self):
        if (self.default_value is not None) and (self.default_value not in self.choices):
            return len(self.choices) + 1
        else:
            return len(self.choices)

    def add_choices(self, new_choices):
        if isinstance(new_choices, list):
            self.choices += new_choices
        else:
            self.choices.append(new_choices)

    def sample(self):
        assert (self.change_chance == 0 or len(self.choices) != 0), 'Chance should be 0 with no choices.'
    
        if random() < self.change_chance:
            value = choice(self.choices)
        else:
            value = self.default_value

        return value

class Layer:
    def __init__(self, default_layer = None, amount_change_chance = None, size_change_chance = None, choice_layer_amount = None, choice_layer_sizes = []):
        assert (((amount_change_chance is None) or (0 <= amount_change_chance <= 1)) and ((size_change_chance is None) or (0 <= size_change_chance <= 1))), 'Chance should be between 0 and 1 if specified.'
        assert (amount_change_chance == None or (default_layer != None and choice_layer_amount != None))
        assert (size_change_chance == None or (default_layer != None and choice_layer_sizes != []))
        assert ((choice_layer_amount == None) == (choice_layer_sizes == [])), 'choice_layer_amount and choice_layer_sizes should either both be set or empty/none.'

        if default_layer != None:
            self.default_layer = default_layer
        else:
            self.amount_change_chance = 1
            self.size_change_chance = 1
            self.default_layer = None

        if choice_layer_amount != []: 
            self.choice_layer_amount = choice_layer_amount
            self.choice_layer_sizes = choice_layer_sizes
        else: 
            self.amount_change_chance = 0
            self.size_change_chance = 1
        
        if default_layer != None and choice_layer_amount != None: 
            self.amount_change_chance = amount_change_chance
            self.size_change_chance = size_change_chance
    
    def get_permutations(self):
        if self.choice_layer_sizes != None:
            prod = 1
            for i in range(1, self.choice_layer_amount):
                prod *= pow(len(self.choice_layer_sizes), i)
            return prod
        else: return 1

    def sample(self):
        n = choice(range(1, self.choice_layer_amount + 1))
        s = [choice(self.choice_layer_sizes) for i in range(0,n)]
        s += [0 for i in range(0, self.choice_layer_amount - n)]
        return s


class ParameterGenerator:
    def __init__(self, unique=False, seed=None):
        assert(seed == None or isinstance(seed, int)),'Seed should be an integer.'

        self.parameters = {}
        if seed != None:
            random_seed(seed)

    def add_value(self, parameter_name, default_value = None, change_chance = None, choices = []):
        assert isinstance(parameter_name, str), 'Parameter name not string object.'
        try: choices = list(choices)
        except: pass
        assert isinstance(choices, (list)),'Choices should be iterable.'

        par = Parameter(default_value, change_chance, choices)
        self.parameters[parameter_name] = par

    def get_permutations(self):
        result = 1
        for key in self.parameters:
            result *= self.parameters[key].get_permutations()
        return result
    
    def add_layer(self, layer_name, default_layer = None, amount_change_chance = None, size_change_chance = None, choice_layer_amount = None, choice_layer_sizes = []):
        assert (amount_change_chance == None or (amount_change_chance <= 1 and 0 <= amount_change_chance and size_change_chance <= 1 and 0 <= size_change_chance)),'Chances should be between 0 and 1.'

        lay = Layer(default_layer, amount_change_chance, size_change_chance, choice_layer_amount, choice_layer_sizes)
        self.parameters[layer_name] = lay

    # This implementation is based on randomness so the time complexity is terrible with a lot of permutations if the argument amount is near the limit.
    def sample(self, amount=1, unique=False):
        assert (len(self.parameters) != 0)
        assert (unique and amount <= self.get_permutations()), 'Cannot generate {} permutations, max permutations is {}.'.format(amount, self.get_permutations())
        
        parameters = []
        for i in range(0, amount):
            parameter_set = {}
            while (parameter_set == {}) or (parameter_set in parameters and unique):
                for key in self.parameters:
                    parameter_set[key] = self.parameters[key].sample()
            parameters.append(parameter_set)
        return parameters
    
    def get_head(self):
        arr = []
        for key in self.parameters:
            if isinstance(self.parameters[key], Layer):
                arr += [(str(key) + '_' + str(i + 1)) for i in range(0, self.parameters[key].choice_layer_amount)]
            else:
                arr.append(str(key))
        return arr

    def as_array(self, parameter):
        arr = []
        for key in parameter:
            assert(key in self.parameters),'Parameter not in parameter generator.'
            if isinstance(self.parameters[key], Layer):
                arr += [str(c) for c in parameter[key]]
            else:
                arr.append(str(parameter[key]))
        return arr
