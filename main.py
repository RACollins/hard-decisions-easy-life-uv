from simulate import LifeSimulation, Person


def main():
    print("Hello from hard-decisions-easy-life-uv!")
    print("I will run my code from here")

    people = [
        Person("Amy", 90),
        Person("Barry", 75),
        Person("Cindy", 64),
        Person("Danny", 45),
        Person("Ethan", 25),
        Person("Frank", 10),
    ]
    simulation = LifeSimulation(people)
    simulation.simlulate_life(2000)


if __name__ == "__main__":
    main()
