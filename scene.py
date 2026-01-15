import numpy as np
import pandas as pd
import plotly.express as px
import sympy as sp


###############
### Classes ###
###############


class Person:
    def __init__(self, name: str, foresight: int):
        self.name = name
        self.foresight = foresight
        self.decision_list = []

    def append_optimal_choice(
        self, choice_array: np.ndarray, decision_number: int, number_of_decisions: int
    ):
        # Get the row index of the highest sum of values in the choice array up to the person's "foresight" index
        highest_value_index = np.argmax(
            np.sum(choice_array[:, : self.foresight], axis=1)
        )

        # Extract this row from the choice array
        optimal_choice = choice_array[highest_value_index, :]

        # Pad initial and final values with nan
        optimal_choice_padded = np.pad(
            optimal_choice,
            (decision_number, number_of_decisions - decision_number),
            mode="constant",
            constant_values=np.nan,
        )

        # Append to decision list
        self.decision_list.append(optimal_choice_padded)

    def convert_decision_list_to_df(self):
        # Swap rows and columns
        decision_array = np.array(self.decision_list)
        decision_array_inverted = decision_array.T
        decision_df = pd.DataFrame(decision_array_inverted)
        self.decision_df = decision_df
        return decision_df

    def calc_life_score(self):
        # Sum columns
        self.decision_df["decision_sum"] = self.decision_df.sum(axis=1)

        # Calculate life score as the cumulative sum of the decision sums
        self.decision_df["life_score"] = self.decision_df["decision_sum"].cumsum()
        return self.decision_df


class LifeSimulation:
    def __init__(
        self,
        people: list[Person],
        genorator: str,
        n_choices: int,
    ):
        self.people = people
        self.genorator = genorator
        self.n_choices = n_choices

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

    def classic_choice_function(self, form: str):
        x = sp.symbols("x")

        m1 = 0
        m2 = 75
        s1 = 5
        s2 = 5

        if form == "now":
            a1 = np.random.uniform(0, 0.01)
            a2 = 0
        elif form == "later":
            a1 = 0
            a2 = np.random.uniform(0, 0.02)

        y = a1 * sp.exp(-((x - m1) ** 2) / (2 * s1**2)) + a2 * sp.exp(
            -((x - m2) ** 2) / (2 * s2**2)
        )

        return {"m1": m1, "m2": m2, "s1": s1, "s2": s2, "a1": a1, "a2": a2, "expr": y}

    def generate_opposites_choice_function(self, form: str):
        x = sp.symbols("x")

        m1 = np.random.uniform(10, 40)
        m2 = np.random.uniform(60, 90)
        s1 = np.random.uniform(1, 5)
        s2 = np.random.uniform(1, 5)

        if form == "bad-good":
            a1 = np.random.uniform(-0.01, 0)
            a2 = np.random.uniform(0, 0.02)
        elif form == "good-bad":
            a1 = np.random.uniform(0, 0.01)
            a2 = np.random.uniform(-0.02, 0)

        y = a1 * sp.exp(-((x - m1) ** 2) / (2 * s1**2)) + a2 * sp.exp(
            -((x - m2) ** 2) / (2 * s2**2)
        )

        return {"m1": m1, "m2": m2, "s1": s1, "s2": s2, "a1": a1, "a2": a2, "expr": y}

    def generate_4_choice_function(self, form: str):
        x = sp.symbols("x")

        m1 = np.random.uniform(10, 40)
        m2 = np.random.uniform(60, 90)
        s1 = np.random.uniform(1, 5)
        s2 = np.random.uniform(1, 5)

        if form == "bad-good":
            a1 = np.random.uniform(-0.01, 0)
            a2 = np.random.uniform(0, 0.02)
        elif form == "good-bad":
            a1 = np.random.uniform(0, 0.01)
            a2 = np.random.uniform(-0.02, 0)
        elif form == "good-good":
            a1 = np.random.uniform(0, 0.01)
            a2 = np.random.uniform(0, 0.02)
        elif form == "bad-bad":
            a1 = np.random.uniform(-0.01, 0)
            a2 = np.random.uniform(-0.02, 0)

        y = a1 * sp.exp(-((x - m1) ** 2) / (2 * s1**2)) + a2 * sp.exp(
            -((x - m2) ** 2) / (2 * s2**2)
        )

        return {"m1": m1, "m2": m2, "s1": s1, "s2": s2, "a1": a1, "a2": a2, "expr": y}

    def generate_random_choice_function(self, seed=None):
        x = sp.symbols("x")

        if seed is not None:
            np.random.seed(seed)

        # Define step boundaries (e.g., every 10 units from 0 to 100)
        step_size = 5
        step_boundaries = np.arange(0, 100, step_size)

        # Generate random values for each step between -0.01 and 0.01
        step_values = np.random.uniform(-0.01, 0.01, size=len(step_boundaries))

        # Create piecewise function using sympy conditions
        conditions = []
        for i, boundary in enumerate(step_boundaries):
            if i == len(step_boundaries) - 1:
                # Last step: from boundary to 100
                conditions.append((step_values[i], sp.And(x >= boundary, x <= 100)))
            else:
                # Regular step: from boundary to next boundary
                next_boundary = step_boundaries[i + 1]
                conditions.append(
                    (step_values[i], sp.And(x >= boundary, x < next_boundary))
                )

        # Create Piecewise expression
        y = sp.Piecewise(*conditions)

        return {
            "step_boundaries": step_boundaries,
            "step_values": step_values,
            "expr": y,
        }

    def generate_choice_array(self, genorator: str, n_choices: int):
        x = sp.symbols("x")

        if genorator == "classic":
            functions = [
                self.classic_choice_function(form=form)
                for form in ["now", "later"] * (n_choices // 2)
            ]
        elif genorator == "opposites":
            functions = [
                self.generate_opposites_choice_function(form=form)
                for form in ["bad-good", "good-bad"] * (n_choices // 2)
            ]
        elif genorator == "4":
            functions = [
                self.generate_4_choice_function(form=form)
                for form in ["bad-good", "good-bad", "good-good", "bad-bad"]
                * (n_choices // 4)
            ]
        elif genorator == "random":
            functions = [
                self.generate_random_choice_function() for choice in range(n_choices)
            ]
        else:
            raise ValueError(f"Invalid generator: {genorator}")

        # Create x values (100 points from 0 to 100)
        x_vals = np.linspace(0, 100, 100)

        # Build NumPy array: n rows (functions) × 100 columns (y values)
        choice_array = np.zeros((n_choices, 100))
        for i, func in enumerate(functions):
            y_func = sp.lambdify(x, func["expr"], "numpy")
            choice_array[i, :] = y_func(x_vals)
        return choice_array

    def simlulate_life(self, number_of_decisions: int):
        for person in self.people:
            for decision_number in range(number_of_decisions):
                choice_array = self.generate_choice_array(
                    genorator=self.genorator, n_choices=self.n_choices
                )
                person.append_optimal_choice(
                    choice_array, decision_number, number_of_decisions
                )
            person.convert_decision_list_to_df()
            person.calc_life_score()
