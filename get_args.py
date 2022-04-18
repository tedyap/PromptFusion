from arguments import get_args

if __name__ == '__main__':

    args = get_args()

    print(args.__dict__)
    model_args, data_args, training_args, _ = args


