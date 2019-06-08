import sys
sys.path.insert(0, r"../../..")
import Utility.ParameterGenerator as Pg
import Utility.BachelorUtilities as Bu

pg = Pg.ParameterGenerator()
pg.add_value('rnn_type', choices=['lstm', 'simplernn'])
pg.add_value('rnn_size', choices=range(1,1000))


a = pg.sample(200, unique=True)
pass