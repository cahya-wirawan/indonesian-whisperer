import unittest
from utils import load_dataset_combination


class LoadDataset(unittest.TestCase):
    def test_dataset_combination_1(self):
        dataset = load_dataset_combination("cahya/newspaper-filtered",
                                           dataset_config_name="kompas-2013",
                                           split="train+test",
                                           streaming=True, use_auth_token=True)
        print(dataset)
        row = next(iter(dataset))
        self.assertEqual(row["title"], "Korban Penembakan di Garut Siuman Setelah Koma 12 Jam")

    def test_dataset_combination_2(self):
        dataset = load_dataset_combination("cahya/newspaper-filtered, ai-research-id/newspaper",
                                           dataset_config_name="kompas-2013, kompas-2013",
                                           split="train+test, train+test",
                                           dataset_data_dir=",",
                                           streaming=True, use_auth_token=True)
        print("dataset:", dataset)
        titles = []
        for i, row in enumerate(dataset):
            titles.append(row["title"])
            if i == 10:
                break
        self.assertEqual(titles[9], "Sneijder Tampik Ada Tawaran dari MU")

    def test_dataset_combination_3(self):
        try:
            dataset = load_dataset_combination("cahya/newspaper-filtered, ai-research-id/newspaper",
                                               dataset_config_name="kompas-2013, kompas-2013",
                                               split="train+test",
                                               dataset_data_dir="",
                                               streaming=True, use_auth_token=True)
        except ValueError as e:
            self.assertEqual(str(e),
                             "dataset_name, dataset_config_name, dataset_data_dir, and split must have the same number of elements")

    def test_dataset_combination_4(self):
        dataset = load_dataset_combination("cahya/newspaper-filtered, ai-research-id/newspaper",
                                           dataset_config_name="kompas-2013, kompas-2013",
                                           split="train+test, train",
                                           dataset_data_dir=",",
                                           streaming=False, use_auth_token=True)
        print("dataset:", dataset)
        self.assertEqual(dataset[9]["title"], "Sneijder Tampik Ada Tawaran dari MU")
        self.assertEqual(len(dataset), 12130)

    def test_dataset_combination_5(self):
        dataset = load_dataset_combination("cahya/newspaper-filtered, ai-research-id/newspaper",
                                           dataset_config_name="kompas-2013, kompas-2013",
                                           split="train+test, train+test",
                                           dataset_data_dir=",",
                                           streaming=True,
                                           shuffle=True,
                                           use_auth_token=True)
        print("dataset:", dataset)
        titles = []
        for i, row in enumerate(dataset):
            titles.append(row["title"])
            if i == 10:
                break
        self.assertEqual(titles[9], "2014, Kemenparekraf Kembangkan Wisata Minat Khusus")

if __name__ == '__main__':
    unittest.main()
