"""
This script processes CSV files containing domain names and updates them with additional characteristics for each domain.
It assumes the presence of an initial 'domain' column and adds the following columns:

- domain_length: The total length of the domain.
- domain_char_count: The count of alphabetical characters in the domain before the first period, which indicates the start of the TLD.
- domain_digit_count: The count of digit characters in the domain before the first period.
- repeated_chars: Indicates with a 1 if there are any consecutive repeating alphabetical characters in the domain before the first period, and 0 otherwise.
- repeated_digits: Indicates with a 1 if there are any consecutive repeating digits in the domain before the first period, and 0 otherwise.
- non_ascii_char_count: The count of non-ASCII Unicode characters in the domain before the first period.
- domain_tld: The top-level domain (TLD), which starts at the first period in the domain name.

The script reads from existing CSV files ('valid-domains.csv' and 'invalid-domains.csv'), processes each row to calculate these values, and then writes the updated data back to the same CSV files.
"""

import csv

def read_data(csv_file_path):
    with open(csv_file_path, mode='r', newline='', encoding='utf-8') as file:  # Specify UTF-8 encoding
        reader = csv.DictReader(file)
        return list(reader)  # Convert iterator to list to reuse the data

def has_repeated_chars(s, char_type):
    return any(s[i] == s[i+1] for i in range(len(s) - 1) if (s[i].isdigit() if char_type == 'digit' else s[i].isalpha()))

def count_non_ascii_chars(s):
    return sum(1 for c in s if ord(c) > 127)

def process_domains(rows, label):
    for row in rows:
        domain = row['domain']
        row['domain_length'] = len(domain)
        
        first_dot_index = domain.find('.')
        base_domain = domain[:first_dot_index] if first_dot_index != -1 else domain
        
        row['domain_char_count'] = sum(c.isalpha() for c in base_domain)
        row['domain_digit_count'] = sum(c.isdigit() for c in base_domain)
        row['non_ascii_char_count'] = count_non_ascii_chars(base_domain)
        row['domain_tld'] = domain[first_dot_index+1:] if first_dot_index != -1 else ''
        row['repeated_chars'] = 1 if has_repeated_chars(base_domain, 'alpha') else 0
        row['repeated_digits'] = 1 if has_repeated_chars(base_domain, 'digit') else 0
        row['domain_label'] = label  # Add domain label based on file type (valid = 1, invalid = 0)

def write_data(csv_file_path, rows):
    fieldnames = ['domain', 'domain_length', 'domain_char_count', 'domain_digit_count', 'repeated_chars', 'repeated_digits', 'non_ascii_char_count', 'domain_tld', 'domain_label']
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:  # Specify UTF-8 encoding
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def main():
    csv_files = [('static/valid-domains.csv', 1), ('static/invalid-domains.csv', 0)]
    for file_path, label in csv_files:
        rows = read_data(file_path)
        process_domains(rows, label)
        write_data(file_path, rows)

if __name__ == "__main__":
    main()
