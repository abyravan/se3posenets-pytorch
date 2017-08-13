#ifndef CSV_UTIL_H
#define CSV_UTIL_H

#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <cassert>

// -----------------------------------------
///
/// \brief find_common_strings - Finds common strings between two vectors of strings (which do not have repeats)
/// \param a - Vector of strings
/// \param b - Vector of strings
/// \return  - Vector of strings that are present in both "a" and "b"
///
std::vector<std::string> find_common_strings(const std::vector<std::string> &a, const std::vector<std::string> &b);

///
/// \brief get_comma_separated_strings - Split a given string into a comma-separated list
/// \param line - string to be separated
/// \return - Vector of comma-separated strings
///
std::vector<std::string> get_comma_separated_strings(const std::string &line);

// -----------------------------------------
///
/// \brief read_csv_labels - Read the comma-separated labels in the first line of the CSV file
/// \param csvfilename  - Path of the file
/// \return - Vector of strings corresponding to the labels of the different columns of the CSV file
///
std::vector<std::string> read_csv_labels(const std::string &csvfilename);

// -----------------------------------------
///
/// \brief read_csv_data - Read the comma-separated data in the CSV file (omitting the first line which is the label line)
/// Only the data for the passed in labels are returned (in same order). Note that these labels have to exist in the CSV file.
/// \param csvfilename  - Path of the file
/// \param labels       - Vector of labels for which we need data
/// \return - Array corresponding to the data for the labels (one vector per row of the CSV file)
///
template <typename T>
std::vector<std::vector<T> > read_csv_data(const std::string &csvfilename, const std::vector<std::string> &labels)
{
    // Get the list of labels for the file
    std::vector<std::string> csv_labels = read_csv_labels(csvfilename);
    std::map<std::string, int> csv_label_map;
    for(int i = 0; i < csv_labels.size(); i++)
        csv_label_map[csv_labels[i]] = i;

    // Ensure that all passed in labels are present in the CSV file. Throw an error if not
    if (find_common_strings(labels, csv_labels).size() != labels.size())
        assert("Some of the passed in labels are not present in CSV file. Check passed in label list!");

    // Open the file
    std::ifstream file(csvfilename);

    // Discard first line
    std::string line;
    std::getline(file, line);
    assert(file.good() && "Could not get a single line from the file!");

    // Iterate over all lines (other than first) and return values corresponding to the labels
    std::vector<std::vector<T> > csv_data;
    while(1)
    {
        // Get a line
        std::getline(file, line);
        if(!file.good()) break;
        std::vector<std::string> vals = get_comma_separated_strings(line);

        // Read into a vector
        std::vector<T> data(labels.size());
        for(int i = 0; i < labels.size(); i++)
            data[i] = (T) atof(vals[csv_label_map[labels[i]]].c_str()); // Get the value for that particular label

        // Add to CSV data
        csv_data.push_back(data);
    }

    // Close file
    file.close();

    // Return the data
    return csv_data;
}

// -----------------------------------------
///
/// \brief read_csv_data - Read the comma-separated data in the CSV file (omitting the first line which is the label line)
/// Only the data for the passed in label is returned. Note that this label has to exist in the CSV file.
/// \param csvfilename  - Path of the file
/// \param label        - Single label for which we need data
/// \return - Array corresponding to the data for the label (one value per row of the CSV file)
///
template <typename T>
std::vector<T> read_csv_data(const std::string &csvfilename, const std::string &label)
{
    // Get data as a vector of vectors
    std::vector<std::string> labels; labels.push_back(label);
    std::vector<std::vector<T> > csv_data = read_csv_data<T>(csvfilename, labels);

    // Squeeze inner vector out
    std::vector<T> csv_data_sq;
    for(int i = 0; i < csv_data.size(); i++)
        csv_data_sq.push_back(csv_data[i][0]);

    // Return squeezed data
    return csv_data_sq;
}

// -----------------------------------------
///
/// \brief read_csv_file - Read the comma-separated data in the CSV file and return both the data & labels
/// \param csvfilename  - Path of the file
/// \param labels       - [RETURN] Vector of labels from the first line of the CSV file
/// \param data         - [RETURN] Vector of data from the CSV file
/// \return - True if success
///
template <typename T>
bool read_csv_file(const std::string &csvfilename, std::vector<std::string> &labels,
                   std::vector<std::vector<T> > &data)
{
    // Get the list of labels for the file
    labels = read_csv_labels(csvfilename);
    if (labels.size() == 0)
    {
        printf("Could not find any labels in the file. Please check the path \n");
        return false;
    }

    // Read the data
    data = read_csv_data<T>(csvfilename, labels);
    if (data.size() == 0)
    {
        printf("No data present in file. \n");
        return false;
    }

    // Return the data
    return true;
}

#endif // CSV_UTIL_H
