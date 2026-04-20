-- Optional context fields for predictions (age, birth_date, notes).

alter table public.predictions
    add column if not exists age integer,
    add column if not exists birth_date date,
    add column if not exists notes text;
