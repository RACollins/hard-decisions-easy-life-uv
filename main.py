from simulate import LifeSimulation, Person


def main():
    print("Hello from hard-decisions-easy-life-uv!")
    print("I will run my code from here")

    people = [
        Person("Amy", 90),
        Person("Barry", 70),
        Person("Cindy", 50),
        Person("Danny", 30),
        Person("Ethan", 10),
    ]
    simulation = LifeSimulation(people)
    simulation.simlulate_life(2000)


if __name__ == "__main__":
    main()
