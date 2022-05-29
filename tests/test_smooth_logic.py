import unittest

from src.smooth_logic.smooth_statement import Chance, Constants


class Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.today_is_sunny = Chance("30%")
        self.tomorrow_will_be_sunny = Chance("70%")
        self.dice = Chance(1 / 6)
        self.coin = Chance(0.5)

    def test_value(self):
        assert self.today_is_sunny.chance == 0.3
        assert self.today_is_sunny.percent == 30

    def test_compare(self):
        assert self.today_is_sunny != self.tomorrow_will_be_sunny
        assert self.today_is_sunny > Chance(0)
        assert self.today_is_sunny < Chance(1)
        assert Chance(0) == Constants.NEVER
        assert Chance(1) == Constants.ALWAYS
        assert Chance(0) == False
        assert Chance(1) == True
        assert Chance(1) != False
        assert Chance(0) != True

    def test_logical_operations(self):
        assert self.today_is_sunny & self.tomorrow_will_be_sunny == Chance(0.21)
        assert self.today_is_sunny | self.tomorrow_will_be_sunny == Chance(0.79)
        assert self.today_is_sunny ^ self.tomorrow_will_be_sunny == Chance(0.58)

        assert self.today_is_sunny & self.coin == 0.15
        assert self.tomorrow_will_be_sunny & self.dice == 0.7 / 6

        assert self.coin & self.dice == self.dice & self.coin
        assert self.coin | self.dice == self.dice | self.coin
        assert self.coin ^ self.dice == self.dice ^ self.coin

        assert self.dice & True == self.dice
        assert self.dice & False == False
        assert self.dice | True == True
        assert self.dice | False == self.dice
        assert self.dice ^ False == self.dice
        assert ~self.dice == 5 / 6

    def test_weight(self):
        assert self.coin * 2 == self.coin.chance**2
        assert self.dice * 2 == self.dice.chance**2
        assert self.dice * 3 == self.dice.chance**3
        assert self.coin * 5 == self.coin.chance**5

    def test_dependency_statements(self):
        assert self.dice & self.dice == self.dice
        assert self.coin | self.coin == self.coin
        # dice_and_coin = self.dice & self.coin
        # assert dice_and_coin & self.dice == dice_and_coin
        # assert dice_and_coin & self.dice & self.coin == dice_and_coin
        # dice_and_coin_and_sunny_on_of_days = dice_and_coin & (self.today_is_sunny | self.tomorrow_will_be_sunny)
        # assert dice_and_coin_and_sunny_on_of_days == 0  # FIXME: Calculate me
        # assert dice_and_coin_and_sunny_on_of_days & self.today_is_sunny < dice_and_coin_and_sunny_on_of_days
