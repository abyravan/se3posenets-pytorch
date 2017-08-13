// Header
#include "csv_util.h"

// General
#include <sstream>

// -----------------------------------------
///
/// \brief find_common_strings - Finds common strings between two vectors of strings (which do not have repeats)
/// \param a - Vector of strings
/// \param b - Vector of strings
/// \return  - Vector of strings that are present in both "a" and "b"
///
std::vector<std::string> find_common_strings(const std::vector<std::string> &a, const std::vector<std::string> &b)
{
    std::vector<std::string> c;
    for(std::size_t i = 0; i < a.size(); i++)
        for(std::size_t j = 0; j < b.size(); j++)
            if (a[i].compare(b[j]) == 0)
                c.push_back(a[i]);

    return c;
}

// -----------------------------------------
///
/// \brief get_comma_separated_strings - Split a given string into a comma-separated list
/// \param line - string to be separated
/// \return - Vector of comma-separated strings
///
std::vector<std::string> get_comma_separated_strings(const std::string &line)
{
    // Get the comma separated values
    std::vector<std::string> vals;
    std::stringstream iss(line);
    while(iss.good())
    {
        std::string val;
        std::getline(iss, val, ',');
        if (val.empty()) continue;
        vals.push_back(val);
    }

    return vals;
}

// -----------------------------------------
///
/// \brief read_csv_labels - Read the comma-separated labels in the first line of the CSV file
/// \param csvfilename  - Path of the file
/// \return - Vector of strings corresponding to the labels of the different columns of the CSV file
///
std::vector<std::string> read_csv_labels(const std::string &csvfilename)
{
    // Open the file
    std::ifstream file(csvfilename);

    // Get the first line and save data to a vector of names
    std::string first_line;
    std::getline(file, first_line);
    assert(file.good() && "Could not get a single line from the file!");
    std::vector<std::string> labels = get_comma_separated_strings(first_line);

    // Close file
    file.close();

    // Return
    return labels;
}
