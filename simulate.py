import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Person:
    def __init__(self, name: str, base_will_power: int):
        self.name = name
        self.base_will_power = base_will_power
        self.current_life_score = 50

    def calc_new_life_score(self, decision_dict: dict[str, int]) -> float:
        good_decision_threshold = (
            self.base_will_power
            + 0.01 * decision_dict["easy_decision_score"]
            + 0.01 * self.current_life_score
        )
        if good_decision_threshold >= decision_dict["hard_decision_score"]:
            self.current_life_score += decision_dict["hard_decision_score"] * 0.01
        else:
            self.current_life_score += decision_dict["easy_decision_score"] * 0.01
        self.current_life_score = np.clip(self.current_life_score, 0, 100)
        return self.current_life_score

    def update_base_will_power(self, new_base_will_power: int):
        self.base_will_power = new_base_will_power


class LifeSimulation:
    def __init__(self, people: list[Person]):
        self.people = people

    def add_person(self, person: Person):
        self.people.append(person)

    def get_people(self) -> list[Person]:
        return self.people

    def get_person_by_name(self, name: str) -> Person | None:
        for person in self.people:
            if person.name == name:
                return person
        return None

    def get_number_of_people(self) -> int:
        return len(self.people)

    def generate_decision_dict(self) -> dict[str, int]:
        hard_decision_score = np.random.randint(0, 100)
        easy_decision_score = np.random.randint(0, 100) * -1
        return {
            "hard_decision_score": hard_decision_score,
            "easy_decision_score": easy_decision_score,
        }

    def simlulate_life(self, number_of_decisions: int):
        life_score_history = {"decision_number": list(range(number_of_decisions))}
        for person in self.people:
            life_score_history[person.name] = []
            for decision_number in range(number_of_decisions):
                decision_dict = self.generate_decision_dict()
                person.calc_new_life_score(decision_dict)
                life_score_history[person.name].append(person.current_life_score)
                if decision_number == number_of_decisions / 2:
                    if person.base_will_power <= 50:
                        person.update_base_will_power(75)
        df = pd.DataFrame(life_score_history)
        df.plot(x="decision_number")
        plt.show()
