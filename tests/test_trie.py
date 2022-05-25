import unittest

from src.trie import Trie


class Empty(unittest.TestCase):
    def setUp(self) -> None:
        self.trie = Trie()

    def test_empty_length(self):
        assert len(self.trie) == 0

class TestGeneration(unittest.TestCase):
    def setUp(self) -> None:
        self.words_list = {'to', 'tea', 'ten', 'ted', 'too', 'a', 'tree'}
        self.trie = Trie.generate(self.words_list)

    def check_len(self):
        assert len(self.trie) == len(self.words_list)

    def check_export_set_of_words(self):
        assert self.words_list == self.trie.export(prefix=None)
        assert {word for word in self.words_list if word.startswith('t')} == self.trie.export(prefix='t', only_path=True)
        assert {word for word in self.words_list if word.startswith('te')} == self.trie.export(prefix='te', only_path=True)

class TestModification(unittest.TestCase):
    def setUp(self) -> None:
        self.words_list = {'to', 'tea', 'ten', 'ted', 'too', 'a', 'tree'}
        self.trie = Trie.generate(self.words_list)

    def check_len(self):
        assert len(self.trie) == len(self.words_list)
        self.trie.append('take')
        self.trie.append('tooth')
        assert len(self.trie) == len(self.words_list) + 2
        self.trie.append('tea')
        assert len(self.trie) == len(self.words_list) + 2

# TODO: Refactor: trie must implement dict interface