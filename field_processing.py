def process_longest_period(field):
    # Replace the "too many to count" value with the 90th percentile value
    many = field.quantile(0.9)
    field[field == -999] = many

    # The value -1 is not specified in the coding and appears to be bogus

    field[field == -1] = float("NaN")
    return field

by_field = {
 20420: process_longest_period,
 20442: process_longest_period,
}
