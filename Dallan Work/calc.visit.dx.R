calc.visit.dx <- function(visit.data, dx.data, dx_bl.data){
  # visit.data must be a data frame with at least RID, EXAMDATE, and USERDATE columns
  # dx.data must be a data frame with at least RID, DIAGNOSIS, USERDATE, USERDATE2, and EXAMDATE columns
  
  # Ensure visit.data is not empty
  if(nrow(visit.data) == 0) 
    return(NULL)
  
  # Ensure dx.data is not empty
  if(nrow(dx.data) == 0)
    return(NULL)
  
  # Ensure dx_bl.data is not empty
  if(nrow(dx_bl.data) == 0)
    return(NULL)
  
  # ---Registry CSV--- #
  
  # Uses EXAMDATE when possible, USERDATE otherwise and sorts by ascending date
  visit.data <- visit.data |>
    group_by(RID) |>
    mutate(
      EXAMDATE = mdy(EXAMDATE), # Ensure date format is valid
      USERDATE = mdy(USERDATE),
      VISITDATE = coalesce(EXAMDATE, USERDATE)
    ) |>
    arrange(RID, VISITDATE) |> # Sort by RID first, then by date
    filter(!is.na(VISITDATE)) |> # Check for NAs
    mutate(
      # baseline is first visit or m0
      bl = first(VISITDATE), 
      # find difference compared to baseline and round to nearest 6 month interval
      month.diff = round(interval(bl, VISITDATE) / months(1) / 6) * 6,
      # Create new viscode labels in the form of "mXX"
      VISCODE = paste0("m", month.diff),
      VISMONTH = month.diff
    ) |>
    ungroup() |>
    select(RID, VISITDATE, VISCODE, VISMONTH) |>
    arrange(RID, VISITDATE)
  
  
  # ---Diagnosis CSV--- #
  
  # Uses EXAMDATE where possible, followed by USERDATE2, and lastly USERDATE
  # Sort ascending
  dx.data <- dx.data |>
    group_by(RID) |>
    mutate(
      EXAMDATE = ymd(EXAMDATE), # Ensure date format is valid
      USERDATE2 = ymd(USERDATE2),
      USERDATE = ymd(USERDATE),
      # Search for the first non-empty value among EXAMDATE, USERDATE2, and USERDATE
      DXDATE = coalesce(EXAMDATE, USERDATE2, USERDATE),
    ) |>
    arrange(RID, DXDATE) |> # Sort by RID first, then by diagnosis date
    filter(!is.na(DXDATE)) |> # Check for NAs
    mutate(
      # baseline is first visit or m0
      bl = first(DXDATE), 
      # find difference compared to baseline and round to nearest 6 month interval
      month.diff = round(interval(bl, DXDATE) / months(1) / 6) * 6,
      # Create new viscode labels in the form of "mXX"
      VISCODE = paste0("m", month.diff),
    ) |>
    ungroup() |>
    select(RID, DXDATE, VISCODE, DIAGNOSIS) |>
    arrange(RID, DXDATE)
  
  
  # ---Merge Diagnosis and DX_bl into Visits--- #
  
  combined.data <- visit.data |> 
    left_join(dx.data) |>
    # Create labels from Diagnoses
    group_by(RID) |>
    mutate(
      DIAGNOSIS = zoo::na.locf(DIAGNOSIS, na.rm = FALSE),
      LAST_DX = last(DIAGNOSIS)
    ) |>
    arrange(RID, VISITDATE) |>
    select(RID, VISITDATE, DIAGNOSIS, VISMONTH, LAST_DX) |>
    distinct(VISITDATE, .keep_all = TRUE)
  
  dx_bl.data <- dx_bl.data |>
    select(RID, DX_bl)
  
  combined.data <- left_join(combined.data, dx_bl.data) |>
    distinct(VISITDATE, .keep_all = TRUE) |>
    mutate(
      DX_bl = case_when(
        DX_bl == 'CN' ~ 0,
        DX_bl == 'SMC' ~ 1,
        DX_bl == 'EMCI' ~ 2,
        DX_bl == 'LMCI' ~ 3,
        DX_bl == 'AD' ~ 4
        )
    )
  
  return(combined.data)
}
