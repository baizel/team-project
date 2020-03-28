import os


def printForGraph(data, key):
    # Header
    print("ImageName,GenAttack,BruteForce,FGSM")
    for entry in data["GenAttack"].keys():
        if data["GenAttack"].get(entry) and data["FGSM"].get(entry) and data["BruteForce"].get(entry):
            gen = data["GenAttack"][entry][key]
            brute = data["BruteForce"][entry][key]
            fg = data["FGSM"][entry][key]
            print("{},{},{},{}".format(entry, gen, brute, fg))


if __name__ == '__main__':
    files = []
    for _, _, filenames in os.walk("attack_examples"):
        files = list(filenames)

    data = {
        "GenAttack": {},
        "BruteForce": {},
        "FGSM": {},
    }

    for i in files:
        attackName, perturbed, original_label, attacked_label, suc, time, original, name = tuple(i.split("_"))
        fulImage = original_label + "_" + original + "_" + name
        data[attackName][fulImage] = {}
        data[attackName][fulImage]["pert"] = perturbed
        data[attackName][fulImage]["time"] = time
        data[attackName][fulImage]["suc"] = suc

    printForGraph(data, "time")
    printForGraph(data, "pert")
    printForGraph(data, "suc")
