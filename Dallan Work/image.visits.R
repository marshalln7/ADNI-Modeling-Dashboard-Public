image.visit <- function(combined.data, image.data){
  # Takes in previously combined data set and merges image data into it
  
  # First ensure new image data is not empty
  if(nrow(image.data) == 0) 
    return(NULL)
  
  # Filter down image data to necessary columns: RID and Acq.Date
  image.data <- image.data |>
    group_by(RID) |>
    mutate(
      # Baseline is first from visit date in combined data
      bl = combined.data |> 
        # filter(RID == cur_group_id()) |> 
        summarize(bl_date = first(VISITDATE)) |> 
        # something doesn't work with this pull
        pull(bl_date), 
      # Find difference compared to visit baseline and round to nearest 6 month interval
      VISMONTH = round(interval(ymd(bl), mdy(Acq.Date)) / months(1) / 6) * 6
    ) |>
    ungroup()
  
  combined.image.data <- combined.data |>
    right_join(image.data, by = c("RID", "VISMONTH")) |>
    rename(MRIDATE = Acq.Date) |>
    mutate(
      MRIDATE = mdy(MRIDATE)
      ) |>
    distinct(RID, VISMONTH, VISITDATE, .keep_all = TRUE) |>
    select(RID, VISITDATE, DIAGNOSIS, DX_bl, VISMONTH, MRIDATE)
    
  return(combined.image.data)
}
