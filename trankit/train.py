from sacred import Experiment
import trankit

ex = Experiment()


@ex.config
def config():
    category = 'customized-ner' # noqa
    max_epoch = 30 # noqa
    save_dir = './save_dir' # noqa
    train_bio_fpath = './train-wo-ryanair.bio' # noqa
    dev_bio_fpath = './dev-w-ryanair.bio' # noqa
    batch_size = 16 # noqa


@ex.automain
def main(category, max_epoch, batch_size, save_dir, train_bio_fpath,
         dev_bio_fpath):

    # initialize a trainer for the task
    trainer = trankit.TPipeline(
        training_config={
            'max_epoch': max_epoch,
            'batch_size': batch_size,
            'category': category,  # pipeline category
            'task': 'ner',  # task name
            'save_dir': save_dir,  # directory to save the trained model
            'train_bio_fpath': train_bio_fpath,  # training data in BIO format
            'dev_bio_fpath': dev_bio_fpath  # training data in BIO format
        }
    )

    # start training
    trainer.train()
