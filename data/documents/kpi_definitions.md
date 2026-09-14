# KPI Definitions

**Average Order Value (AOV)** = AVG(invoices.Total) across all invoices, or
optionally filtered to a specific date range using `invoices.InvoiceDate`.

**Revenue by Country** = SUM(invoices.Total) grouped by
`invoices.BillingCountry`. This is the standard geographic revenue
breakdown used for regional performance questions.

**Catalog Coverage** = the percentage of tracks in the `tracks` table that
appear at least once in `invoice_items` (i.e. tracks that have ever sold).
A low catalog coverage percentage indicates a "long tail" catalog where most
revenue comes from a small number of tracks.

**Playlist Engagement** = COUNT(playlist_track rows) grouped by
`playlists.PlaylistId`, joined to `playlists.Name` for display. This
measures how many tracks have been added to a playlist, not how many times
those tracks were purchased.

**Data Freshness Note**: `invoices.InvoiceDate` is stored as a text
timestamp (e.g. `2009-01-01 00:00:00`). Date range filters should use SQLite
string comparison or `date()`/`strftime()` functions, not a native DATE
type.
