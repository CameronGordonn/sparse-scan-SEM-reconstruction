"""Tune TV lambdas on val and evaluate methods on the frozen test set. See semrecon.evaluate."""

from semrecon.evaluate import main, parse_args

if __name__ == "__main__":
    main(parse_args())
