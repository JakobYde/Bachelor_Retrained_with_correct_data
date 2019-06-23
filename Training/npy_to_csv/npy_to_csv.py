import sys
sys.path.insert(0, r"../..")
import Utility.bachelor_utilities as Bu

npy_path = 'D:\WindowsFolders\Code\Data\BachelorFixedData/'
csv_path = 'D:\WindowsFolders\Code\Data\BachelorFixedData/csv_files/'

Bu.npy_to_csv(npy_path, csv_path)