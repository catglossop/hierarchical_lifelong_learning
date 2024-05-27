from google.cloud import storage


def move_blobs(bucket_name):
    """Moves a blob from one bucket to another with a new name."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"
    # The ID of your GCS object
    # blob_name = "your-object-name"
    # The ID of the bucket to move the object to
    # destination_bucket_name = "destination-bucket-name"
    # The ID of your new GCS object (optional)
    # destination_blob_name = "destination-object-name"

    storage_client = storage.Client()

    source_bucket = storage_client.bucket(bucket_name)

    source_blobs = storage_client.list_blobs(bucket_name)
    destination_bucket = storage_client.bucket(bucket_name)
    destination_generation_match_precondition = 0 
    for blob in source_blobs: 
        if blob.name.split("/")[0] == "lifelong":
            new_folder = ("_").join(blob.name.split("/")[1].split("_")[2:])
            new_blob_name = "lifelong_datasets/" + new_folder + "/lifelong_data/" + ("/").join(blob.name.split("/")[2:])
            print(new_blob_name)

            blob_copy = source_bucket.copy_blob(blob, destination_bucket, new_blob_name, if_generation_match=destination_generation_match_precondition,)

        #source_bucket.delete_blob(blob)

if __name__ == "__main__":

    move_blobs("catg_central2")
