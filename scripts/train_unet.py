"""Thin wrapper: see semrecon.train_unet."""

from semrecon.train_unet import main, parse_args

if __name__ == "__main__":
    main(parse_args())
