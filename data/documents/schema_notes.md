# Schema Notes (Chinook sample database)

The Chinook database models a digital media store. Its core join path for
sales analysis is:

`customers -> invoices -> invoice_items -> tracks -> albums -> artists`

and separately:

`tracks -> genres`, `tracks -> media_types`, and `playlists <-> tracks` via
the `playlist_track` bridge table.

`invoice_items.UnitPrice * invoice_items.Quantity` is the line-item revenue
for a single purchased track; `invoices.Total` is the pre-aggregated total
for the whole invoice and should already equal the sum of its line items.

`employees.ReportsTo` links each employee to their manager's
`EmployeeId`; the top of the hierarchy has `ReportsTo IS NULL`.

There are no `DROP`, `DELETE`, `UPDATE`, or `INSERT` privileges granted to
the query agent in this project - every generated query is validated to be
a read-only `SELECT` (or `WITH ... SELECT`) statement before it is executed
against the database.
