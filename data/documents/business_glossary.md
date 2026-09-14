# Business Glossary

**Total Spending / Customer Spend** refers to the sum of the `Total` column
in the `invoices` table for a given `CustomerId`. This represents the
lifetime revenue collected from that customer across all their invoices.

**Top Customer** is a customer ranked by total spending (sum of
`invoices.Total`) in descending order. "Top N customers" queries should
group `invoices` by `CustomerId`, sum `Total`, join back to `customers` for
display fields such as `FirstName`/`LastName`/`Company`, and order by the
summed total descending with a `LIMIT`.

**Active Customer** is a customer who has at least one row in `invoices`
within the analysis window. Customers with zero invoices should generally be
excluded from spending-based rankings unless the question explicitly asks
about all customers including those with no purchases.

**Best Selling Track / Top Track** is a track ranked by the number of times
it appears across all `invoice_items` rows (i.e. the number of times it has
been purchased), not by how many playlists it appears in.

**Genre Popularity** is measured by counting `invoice_items` joined through
`tracks` to `genres`, not by counting tracks per genre in the catalog alone -
a genre can have many tracks in the catalog but few actual sales.

**Employee Hierarchy** is represented by the `employees.ReportsTo` column,
which is a self-referencing foreign key to `employees.EmployeeId`.
