# isort: skip_file
from engine import VisualEntailmentModelEngine


def main():
    engine = VisualEntailmentModelEngine()
    if engine.training_args.do_train:
        engine.train()
    elif engine.training_args.do_predict:
        engine.predict()
    else:
        raise ValueError('make one of [do_train, do_predict] true')


if __name__ == "__main__":
    main()
