import argparse
import os




def main():
    parser = argparse.ArgumentParser(description='Run trained models on given input file')
    parser.add_argument('--i', metavar='N', type=str, nargs='?',
                        help='Input filename')
    # parser.add_argument('--sum', dest='accumulate', action='store_const',
    #                     const=sum, default=max,
    #                     help='sum the integers (default: find the max)')

    args = parser.parse_args()
    
    eval_filename = args.i
    
    if not eval_filename.lower().endswith('.csv'):
        raise ValueError('The input file should be in the .csv file format')

    if os.path.exists(eval_filename):
        
    else:
        raise FileNotFoundError('Cannot locate the file at the given path. Please check the provided filepath')

if __name__ == "__main__":
    main()

