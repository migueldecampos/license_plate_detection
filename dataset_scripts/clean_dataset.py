import csv
import os


def find_problematic_labels(root):
    errors = dict()
    for mode in ("train", "test", "val"):
        errors[mode] = list()
        labels = list(sorted(os.listdir(os.path.join(root, "labels", mode))))
        for label in labels:
            file_path = os.path.join(root, "labels", mode, label)
            with open(file_path, newline="\n") as csvfile:
                reader = csv.reader(csvfile, delimiter=" ")
                rows = list(reader)
                if not rows:
                    # print("ERROR (no rows) {} {}\n{}\n".format(mode, label, file_path))
                    errors[mode].append(label)
                else:
                    for row in reader:
                        if len(row) != 5:
                            # print(
                            #    "ERROR (incomplete row) {} {}\n{}\n".format(
                            #        mode, label, file_path
                            #    )
                            # )
                            errors[mode].append(label)
                            break
                        width = float(row[3])
                        height = float(row[4])
                        if not width or not height:
                            # print(
                            #    "ERROR (empty box) {} {}\n{}\n".format(
                            #        mode, label, file_path
                            #    )
                            # )
                            errors[mode].append(label)
                            break
    return errors


def delete_image_and_label(root, errors):
    """
    This function receives a list of labels, as outputed by find_problematic_labels(),
    and deletes labels and corresponding images.
    """
    path_list = list()
    for mode in errors:
        for label in errors[mode]:
            data_id = label[:-4]
            image_path = os.path.join(root, "images", mode, data_id + ".jpg")
            label_path = os.path.join(root, "labels", mode, data_id + ".txt")
            path_list.append(image_path)
            path_list.append(label_path)
            print(image_path)
            print(os.path.isfile(image_path))
            print(label_path)
            print(os.path.isfile(label_path))

    print()
    print(" ".join(path_list))
    print()


if __name__ == "__main__":
    root = "../../data/LargeLicensePlateDetectionDataset"
    errors = find_problematic_labels(root=root)

    delete_image_and_label(root=root, errors=errors)
