-- Table privileges for authenticated role (parity with public.profiles).
-- RLS policies alone do not grant table-level access on fresh Supabase projects.

grant select, insert on table public.predictions to authenticated;
