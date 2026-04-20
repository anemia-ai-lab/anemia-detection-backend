-- Email vive solo en auth.users; la API expone email desde GoTrue al leer perfil.
-- profile_completed: el usuario puede completar datos más tarde (PATCH).

alter table public.profiles
    add column if not exists profile_completed boolean not null default false;

alter table public.profiles
    drop column if exists email;
